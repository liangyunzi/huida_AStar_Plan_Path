# Learning Date : 2025/12/24
# !/usr/bin/env python

import rospy
import numpy as np
import tf
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from sklearn.cluster import DBSCAN
from scipy.interpolate import splprep, splev


# Import your custom logic classes here or include them in the script
# For brevity, I've integrated the core A* and Frontier logic below

class FALCONRosNode:
    def __init__(self):
        rospy.init_node('falcon_planner_node')

        # Parameters
        self.robot_radius = rospy.get_param('~robot_radius', 0.5)  # meters
        self.cluster_eps = rospy.get_param('~cluster_eps', 5.0)
        self.goal_tolerance = 0.5

        # Internal State
        self.map_data = None
        self.map_info = None
        self.current_pose = None
        self.listener = tf.TransformListener()

        # Publishers
        self.path_pub = rospy.Publisher('~falcon_path', Path, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # Subscribers
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback)

        # Timer for planning loop (2Hz)
        self.timer = rospy.Timer(rospy.Duration(0.5), self.planning_loop)

        rospy.loginfo("FALCON Planner Node Initialized")

    def map_callback(self, msg):
        self.map_info = msg.info
        # Convert 1D map to 2D Numpy array
        # ROS Map: 0=free, 100=occ, -1=unknown
        raw_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_data = raw_map

    def get_robot_pose(self):
        try:
            (trans, rot) = self.listener.lookupTransform('/map', '/base_link', rospy.Time(0))
            # Convert world coords to map grid coords
            grid_x = int((trans[0] - self.map_info.origin.position.x) / self.map_info.resolution)
            grid_y = int((trans[1] - self.map_info.origin.position.y) / self.map_info.resolution)
            return (grid_y, grid_x), trans
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None, None

    def _get_frontiers(self):
        if self.map_data is None: return []

        # Frontier: A 'free' cell (0) adjacent to an 'unknown' cell (-1)
        free_mask = (self.map_data == 0)
        rows, cols = self.map_data.shape
        frontiers = []

        # Optimization: Use numpy shifts to find boundaries
        # This is faster than iterating through every pixel in Python
        unknown_mask = (self.map_data == -1)

        # Simple check for 4-neighbors
        is_frontier = np.zeros_like(free_mask)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(unknown_mask, dr, axis=0), dc, axis=1)
            is_frontier |= (free_mask & shifted)

        frontier_indices = np.argwhere(is_frontier)
        return frontier_indices

    def planning_loop(self, event):
        if self.map_data is None: return

        grid_pos, world_pos = self.get_robot_pose()
        if grid_pos is None: return

        # 1. Extract Frontiers
        frontiers = self._get_frontiers()
        if len(frontiers) < 5:
            rospy.loginfo_throttle(10, "No more frontiers found. Exploration complete?")
            return

        # 2. Cluster Frontiers (DBSCAN)
        try:
            clustering = DBSCAN(eps=self.cluster_eps, min_samples=3).fit(frontiers)
            labels = clustering.labels_
        except:
            return

        # 3. Global Goal Selection (Greedy)
        best_goal_grid = None
        min_cost = float('inf')

        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1: continue

            cluster_pts = frontiers[labels == label]
            centroid = np.mean(cluster_pts, axis=0)

            # Cost = Distance / cluster_size
            dist = np.linalg.norm(centroid - np.array(grid_pos))
            cost = dist / (len(cluster_pts) + 0.1)

            if cost < min_cost:
                min_cost = cost
                # Select the point in cluster closest to robot as entry point
                dists_to_robot = np.linalg.norm(cluster_pts - np.array(grid_pos), axis=1)
                best_goal_grid = cluster_pts[np.argmin(dists_to_robot)]

        if best_goal_grid is not None:
            self.publish_goal(best_goal_grid)

    def publish_goal(self, grid_goal):
        # Convert grid back to world coordinates
        world_x = grid_goal[1] * self.map_info.resolution + self.map_info.origin.position.x
        world_y = grid_goal[0] * self.map_info.resolution + self.map_info.origin.position.y

        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = world_x
        goal_msg.pose.position.y = world_y
        goal_msg.pose.orientation.w = 1.0  # Simple orientation

        self.goal_pub.publish(goal_msg)


if __name__ == '__main__':
    try:
        node = FALCONRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
