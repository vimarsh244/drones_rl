import gymnasium as gym
import numpy as np
import torch as th
from torch import nn
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm

from turtlebot_nav_env import Turtlebot_Nav_Env

class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, feature_dim=256):
        super(ICM, self).__init__()
        self.feature_dim = feature_dim
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
    def forward(self, state, next_state, action):
        state_feat = self.feature_extractor(state)
        next_state_feat = self.feature_extractor(next_state)
        
        # Inverse Model
        inverse_input = th.cat([state_feat, next_state_feat], dim=1)
        pred_action = self.inverse_model(inverse_input)
        
        # Forward Model
        forward_input = th.cat([state_feat, action], dim=1)
        pred_next_state_feat = self.forward_model(forward_input)
        
        return pred_action, pred_next_state_feat, next_state_feat

class PPOWithICMCallbackOld(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(PPOWithICMCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.icm = None
        self.icm_optimizer = None

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        obs_dim = self.model.observation_space.shape[0]
        action_dim = self.model.action_space.shape[0]
        self.icm = ICM(obs_dim, action_dim).to(self.model.device)
        self.icm_optimizer = th.optim.Adam(self.icm.parameters(), lr=1e-3)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print("Saving Model....")
            self.model.save(self.save_path)

        # ICM update
        rollout_buffer = self.model.rollout_buffer

        if rollout_buffer.full:
            obs = rollout_buffer.observations
            actions = rollout_buffer.actions
            next_obs = rollout_buffer.observations[1:].copy()  # Shift observations by 1
            next_obs = np.vstack([next_obs, rollout_buffer.observations[-1]])  # Add the last observation

            obs = th.as_tensor(obs).float().to(self.model.device)
            actions = th.as_tensor(actions).float().to(self.model.device)
            next_obs = th.as_tensor(next_obs).float().to(self.model.device)

            pred_action, pred_next_state_feat, next_state_feat = self.icm(obs, next_obs, actions)

            inverse_loss = nn.MSELoss()(pred_action, actions)
            forward_loss = nn.MSELoss()(pred_next_state_feat, next_state_feat.detach())
            icm_loss = inverse_loss + forward_loss

            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            self.icm_optimizer.step()

            # Compute intrinsic reward
            intrinsic_reward = forward_loss.detach().cpu().numpy()
            print(f"Intrinsic reward : {intrinsic_reward}")

            # Add intrinsic reward to extrinsic reward
            rollout_buffer.rewards += 0.01 * intrinsic_reward  # You may need to adjust this scaling factor

        return True

class PPOWithICMCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(PPOWithICMCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.icm = None
        self.icm_optimizer = None
        self.update_count = 0

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        obs_dim = self.model.observation_space.shape[0]
        action_dim = self.model.action_space.shape[0]
        self.icm = ICM(obs_dim, action_dim).to(self.model.device)
        self.icm_optimizer = th.optim.Adam(self.icm.parameters(), lr=1e-3)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(f"Step {self.n_calls}: Saving Model....")
            self.model.save(self.save_path)

        # ICM update
        rollout_buffer = self.model.rollout_buffer

        if rollout_buffer.full:
            self.update_count += 1
            print(f"\n--- ICM Update #{self.update_count} ---")
            
            obs = rollout_buffer.observations
            actions = rollout_buffer.actions
            next_obs = rollout_buffer.observations[1:].copy()
            next_obs = np.vstack([next_obs, rollout_buffer.observations[-1]])

            obs = th.as_tensor(obs).float().to(self.model.device)
            actions = th.as_tensor(actions).float().to(self.model.device)
            next_obs = th.as_tensor(next_obs).float().to(self.model.device)

            pred_action, pred_next_state_feat, next_state_feat = self.icm(obs, next_obs, actions)

            inverse_loss = nn.MSELoss()(pred_action, actions)
            forward_loss = nn.MSELoss()(pred_next_state_feat, next_state_feat.detach())
            icm_loss = inverse_loss + forward_loss

            self.icm_optimizer.zero_grad()
            icm_loss.backward()
            self.icm_optimizer.step()

            intrinsic_reward = forward_loss.detach().cpu().numpy()

            # print(f"Inverse Loss: {inverse_loss.item():.6f}")
            # print(f"Forward Loss: {forward_loss.item():.6f}")
            # print(f"ICM Loss: {icm_loss.item():.6f}")
            # print(f"Intrinsic Reward (mean): {np.mean(intrinsic_reward):.6f}")
            # print(f"Intrinsic Reward (min): {np.min(intrinsic_reward):.6f}")
            # print(f"Intrinsic Reward (max): {np.max(intrinsic_reward):.6f}")

            scaling_factor = 0.2
            rollout_buffer.rewards += scaling_factor * intrinsic_reward
            # print(f"Scaling Factor: {scaling_factor}")
            # print(f"Updated Rewards (mean): {np.mean(rollout_buffer.rewards):.6f}")
            # print(f"Updated Rewards (min): {np.min(rollout_buffer.rewards):.6f}")
            # print(f"Updated Rewards (max): {np.max(rollout_buffer.rewards):.6f}")

        return True


# Parallel environments
env = Turtlebot_Nav_Env()
callback = PPOWithICMCallback(10000, "tmp/")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1100000, callback=callback)
model.save("ppo_icm_tb3")

