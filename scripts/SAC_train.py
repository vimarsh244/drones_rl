import gymnasium as gym

from stable_baselines3 import SAC

from turtlebot_nav_env import Turtlebot_Nav_Env

env = Turtlebot_Nav_Env()

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1125000, log_interval=4)
model.save("sac_tb3")

del model # remove to demonstrate saving and loading

