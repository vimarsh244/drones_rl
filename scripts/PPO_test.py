import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import os
from turtlebot_nav_env import Turtlebot_Nav_Env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

env = Turtlebot_Nav_Env()

model = PPO.load("ppo_tb3")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncate, info = env.step(action)