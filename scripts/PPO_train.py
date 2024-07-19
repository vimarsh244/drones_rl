import gymnasium as gym

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
import os
from turtlebot_nav_env import Turtlebot_Nav_Env
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        print("Saving Model....")
        self.model.save(self.save_path)

        return True
    

# Parallel environments
env = Turtlebot_Nav_Env()
callback = SaveOnBestTrainingRewardCallback(10000, "tmp/")

# model = PPO("MlpPolicy", env, verbose=1)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1100000)
model.save("ddpg_tb3")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_tb3")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, truncate, info = env.step(action)
