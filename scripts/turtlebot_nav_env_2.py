import numpy as np
import time 
from control import Controller
import rospy
from utils.rewards import compute_rewards
from gymnasium.spaces import Box, Dict
import gymnasium

class Turtlebot_Nav_Env(gymnasium.Env):
    def __init__(self):
        self.controller = Controller()
        self.reward = 0

        # Assume lidar data is 9-dimensional and camera data is (height, width, channels)
        self.observation_space = Dict({
            "lidar": Box(low=0, high=5.0, shape=(9,), dtype=np.float32),
            "camera": Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        })
        self.action_space = Box(-1, 1, shape=(2,))

        self.steps = 0
        self.eps = 0
        self.ep_time = time.time()
        self.max_time = 2500

    def step(self, action):
        self.controller.execute_action(action)
        state = self.controller.get_state()  # This should now return both lidar and camera data
        done = self.controller.is_done() 
        truncated = (self.steps >= self.max_time)
        
        # Pass only the lidar data to compute_rewards
        reward = compute_rewards(state['lidar'], self.controller.occupancy_grid, self.controller.w)
        
        self.reward += reward
        self.steps += 1
        
        return state, reward, done, truncated, {"TimeLimit.truncated": truncated}

    def reset(self, seed=None):
        self.ep_time = time.time() - self.ep_time
        status = f"Episode : {self.eps}  ;  Time : {self.ep_time}  ;  Reward : {self.reward}"
        self.controller.status_pub.publish(status)

        self.controller.reset_simulation()
        self.controller.setup_launcher()
        rospy.sleep(1)
        self.reward = 0
        self.steps = 0

        self.eps += 1
        self.ep_time = time.time()
        self.max_time += 5
        self.max_time = min(self.max_time, 3000)
        
        initial_state = self.controller.get_state()
        
        return initial_state, None

    def close(self):
        self.controller.launcher.shutdown()