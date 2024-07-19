import rospy
import numpy
from utils.ocp_grid import *
#!/usr/bin/env python3
import message_filters
import rospy
import math
import time
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan, Range
from gazebo_msgs.srv import DeleteModel
from std_srvs.srv import Empty
from std_msgs.msg import String, Int32MultiArray
import tf.transformations
import roslaunch
import rospy
import numpy as np
from tf.transformations import euler_from_quaternion
import cv2
import threading

class Mapping():
    def __init__(self) -> None:
        rospy.init_node('ocp_grid')

        self.rate = rospy.Rate(30)

        self.current_state = Odometry()

        # Example usage
        self.grid_size = (112, 92)
        self.grid_resolution = 0.05  # Each cell represents 0.05 meters
        self.occupancy_grid = np.full(self.grid_size, 5, dtype=np.int8)
        # self.prev_occupancy_grid = np.full(self.grid_size, 0.5)  # Initialize with unknown (0.5)
        # self.entropy_grid = self.calculate_entropy(self.occupancy_grid)
        self.robot_position = (71, 48)  # Robot starts at the center of the grid

        self.msg = OccupancyGrid()

        #self.odom_sub = rospy.Subscriber("/odom", Odometry, self.state_cb, queue_size=1)

        self.reset_sub= rospy.Subscriber("/reset_env", String, self.reset_cb, queue_size=1)

        #self.lidar_scan_sub = rospy.Subscriber("/scan", LaserScan, self.scan_cb, queue_size=1)

        self.grid_pub = rospy.Publisher("/ocp_grid", OccupancyGrid, queue_size=10)

        self.odom_sub = message_filters.Subscriber("/odom", Odometry)
        self.lidar_scan_sub = message_filters.Subscriber("/scan", LaserScan)
        
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.lidar_scan_sub],
            queue_size=1,
            slop=0.1,
            allow_headerless=True
        )
        ts.registerCallback(self.callback)

        # self.lock = threading.Lock()
        # self.update_thread = threading.Thread(target=self.update_loop)
        # self.update_thread.start()

        self.x = None
        self.y = None
        self.yaw = 0



    # def update_loop(self):
    #     while not rospy.is_shutdown():
    #         with self.lock:
    #             if hasattr(self, 'latest_scan') and hasattr(self, 'latest_odom'):
    #                 self.occupancy_grid = update_occupancy_grid(
    #                     self.occupancy_grid,
    #                     self.latest_scan.ranges,
    #                     self.robot_position,
    #                     self.grid_resolution,
    #                     self.yaw
    #                 )
    #         rospy.sleep(0.001)  # Adjust this value as needed


    # def callback(self, odom_msg, scan_msg):
    #     with self.lock:
    #         self.latest_odom = odom_msg
    #         self.latest_scan = scan_msg
    #     self.state_cb(odom_msg)
    #     self.scan_cb(scan_msg)


    def callback(self, odom_msg, scan_msg):
        self.state_cb(odom_msg)
        self.scan_cb(scan_msg)


    def scan_cb(self, msg):
        ranges = [5.0 if math.isinf(r) else r for r in msg.ranges]

        self.occupancy_grid = update_occupancy_grid(self.occupancy_grid, ranges, self.robot_position, self.grid_resolution, self.yaw)

        grid = OccupancyGrid()
        arr = self.occupancy_grid

        # uint_img = np.array(self.occupancy_grid*25.5).astype('uint8')
        # grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
        # bigger = cv2.resize(grayImage, (112*3, 92*3))
        # cv2.imshow("Image1", bigger)
        # cv2.waitKey(1)
        # print("Showing....")

        grid.data = arr.ravel()
        grid.info.height = arr.shape[0]
        grid.info.width = arr.shape[1]
        self.grid_pub.publish(grid)


    def state_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll, pitch, self.yaw) = euler_from_quaternion([rot_q.x, rot_q.y, rot_q.z, rot_q.w])
        if self.yaw < 0:
            self.yaw += 2 * np.pi

        # print(self.yaw)

        self.robot_position = (int((self.x + 2.3)/0.05), int((self.y + 2.3)/0.05))


    def reset_cb(self, msg):
        if str(msg.data) == 'r':
            print("Resetting")
            self.occupancy_grid = np.full(self.grid_size, 5)
            self.prev_occupancy_grid = np.full(self.grid_size, 5)  # Initialize with unknown (0.5)
            self.robot_position = (71, 48)



mapper = Mapping()
rospy.spin()