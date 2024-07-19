import numpy as np
import time 
from control import Controller
import rospy
from utils.rewards import compute_rewards
from gymnasium.spaces import Box
import gymnasium

class Turtlebot_Nav_Env(gymnasium.Env):
    def __init__(self):
        # Initialize control node
        self.controller = Controller()
        self.reward = 0

        self.observation_space = Box(0, 5.0, shape=(9,))
        self.action_space = Box(-1, 1, shape=(2,))

        self.steps = 0
        self.eps = 0
        self.ep_time = time.time()
        self.max_time = 2500

    def step(self, action):
        self.controller.execute_action(action)
        state = self.controller.get_state()
        done = self.controller.is_done() 
        truncated = (self.steps >= self.max_time)
        reward = compute_rewards(state, self.controller.occupancy_grid, self.controller.w)
        self.reward += reward
        # self.controller.status_pub.publish("Steps : "+str(self.steps) + )
        # self.controller.reward_pub.publish(str(reward))

        self.steps += 1

        return state, reward, done, truncated, {"TimeLimit.truncated": truncated}


    def render(self):
        pass


    def reset(self, seed=None):
        # self.controller.launcher.shutdown()
        # self.controller.process.stop()
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

        return self.controller.get_state(), None

    
    def close(self):
        self.controller.launcher.shutdown()
