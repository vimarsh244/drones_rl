#!/usr/bin/env python3

import rospy
import math
import time
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Range
from gazebo_msgs.srv import DeleteModel
from std_srvs.srv import Empty
from std_msgs.msg import String, Int32MultiArray
import tf
import tf.transformations
from mavros_msgs.srv import CommandLong
import roslaunch
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
import cv2
from utils.ocp_grid import *


class Controller:
    def __init__(self):
        rospy.init_node('Controller')

        self.rate = rospy.Rate(30)

        self.current_state = Odometry()
        self.map = None
        self.prev_explored = 0
        self.explored = 0
        self.vel_msg = Twist()
        self.vel_msg.linear.x = 0
        self.vel_msg.angular.z = 0
        self.explore_thresh = 95
        self.process = None

        self.state = None

        # Example usage
        self.grid_size = (112, 92)
        self.grid_resolution = 0.05  # Each cell represents 0.05 meters
        self.occupancy_grid = np.full(self.grid_size, 0.5)
        # self.prev_occupancy_grid = np.full(self.grid_size, 0.5)  # Initialize with unknown (0.5)
        # self.entropy_grid = self.calculate_entropy(self.occupancy_grid)
        # self.robot_position = (71, 48)  # Robot starts at the center of the grid

        self.odom_cb = rospy.Subscriber("/odom", Odometry, self.state_cb, queue_size=2)

        self.lidar_scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_cb, queue_size=2)

        self.grid_sub = rospy.Subscriber("/ocp_grid", OccupancyGrid, self.grid_cb, queue_size=2)

        self.vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        self.status_pub = rospy.Publisher("/train_status", String, queue_size=10)

        self.reward_pub = rospy.Publisher("/train_rewards", String, queue_size=10)

        self.reset_pub = rospy.Publisher("/reset_env", String, queue_size=10)

        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.x = None
        self.y = None
        self.yaw = 0
        self.w = 0

        # self.launcher = roslaunch.scriptapi.ROSLaunch()
        # self.launcher.start()
        # self.node = roslaunch.core.Node("gmapping", "slam_gmapping", name="slam_gmapping", output='log')


        self.setup_launcher()
        for i in range(100):
            rospy.Rate.sleep(self.rate)


    def setup_launcher(self):
        # self.uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(self.uuid)
        # self.launcher = roslaunch.parent.ROSLaunchParent(self.uuid, ["/home/csir/ritwik/ritwik_ws/src/multi_critic_rl/launch/rtab_test.launch"])
        # self.launcher.start()
        #self.process = self.launcher.launch(self.node)

        self.explored = 0
        self.map = None
        self.prev_explored = 0
        self.occupancy_grid = np.full(self.grid_size, 0.5)
        self.prev_occupancy_grid = np.full(self.grid_size, 0.5)  # Initialize with unknown (0.5)
        self.robot_position = (71, 48)
        # rospy.wait_for_message("/rtabmap/grid_map", OccupancyGrid)
        # self.map_sub = rospy.Subscriber("/rtabmap/grid_map", OccupancyGrid, self.map_cb, queue_size=10)
        self.reset_pub.publish("r")
        for i in range(200):
            rospy.Rate.sleep(self.rate)


    def scan_cb(self, msg):
        ranges = [5.0 if math.isinf(r) else r for r in msg.ranges]
        state = [min(ranges[340:] + ranges[:20])]
        for i in range(8):
            state.append(min(ranges[20 + 40 * i: 60 + 40 * i]))

        self.state = state

        # Update the occupancy grid with LiDAR data
        # self.prev_occupancy_grid = self.occupancy_grid
        #self.occupancy_grid = update_occupancy_grid(self.occupancy_grid, ranges, self.robot_position, self.grid_resolution, self.yaw)
        # uint_img = np.array(self.occupancy_grid*255).astype('uint8')
        # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
        # bigger = cv2.resize(grayImage, (112*3, 92*3))
        # cv2.imshow("Image", bigger)
        # cv2.waitKey(1)
        # self.explored = (float(np.count_nonzero(self.occupancy_grid == 1))/100)
        #self.prev_explored = self.explored


    def get_state(self):
        return self.state


    def state_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll, pitch, self.yaw) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        if self.yaw < 0:
            self.yaw += 2 * np.pi

        # self.robot_position = (int((self.x + 2.3)/0.05), int((self.y + 2.3)/0.05))
        self.w = self.vel_msg.angular.z

    
    def is_done(self):
        return (self.explored >= self.explore_thresh)


    def map_cb(self, msg):
        self.map = np.array(msg.data)
        #self.prev_explored = self.explored
        self.explored = float(np.count_nonzero(self.map == 0) / 100)

    
    def execute_action(self, act):
        self.vel_msg.linear.x = act[0] * 0.25
        self.vel_msg.angular.z = act[1] * 0.524
        self.vel_pub.publish(self.vel_msg)


    def grid_cb(self, msg):
        data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.occupancy_grid = data/10
        # uint_img = np.array(self.occupancy_grid*255).astype('uint8')
        # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
        # bigger = cv2.resize(grayImage, (112*3, 92*3))
        # cv2.imshow("Image1", bigger)
        # cv2.waitKey(1)
        # print("Showing....")


