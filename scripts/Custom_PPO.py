import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3 import PPO

import os
import numpy as np

import gymnasium as gym

from turtlebot_nav_env_2 import Turtlebot_Nav_Env
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import VecTransposeImage

import torch as th
import torch.nn as nn
import numpy as np

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=256)
        
        n_input_channels = observation_space["lidar"].shape[-1]  # Should be 9
        self.lidar_net = nn.Sequential(
            nn.Linear(n_input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        
        n_input_channels_cam = observation_space["camera"].shape[0]  # Should be 3
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels_cam, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space["camera"].sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())
        
    def forward(self, observations):
        # Print shapes for debugging
        print(f"Lidar shape: {observations['lidar'].shape}")
        print(f"Camera shape: {observations['camera'].shape}")

        # Handle both batched and unbatched inputs
        if observations['lidar'].dim() == 2:
            # Batched input
            lidar_tensor = observations['lidar'].float()
            camera_tensor = observations['camera'].float()
        elif observations['lidar'].dim() == 1:
            # Unbatched input
            lidar_tensor = observations['lidar'].float().unsqueeze(0)
            camera_tensor = observations['camera'].float().unsqueeze(0)
        else:
            raise ValueError(f"Unexpected lidar tensor dimension: {observations['lidar'].dim()}")

        lidar_features = self.lidar_net(lidar_tensor)
        cnn_features = self.linear(self.cnn(camera_tensor))
        
        # Ensure both features have the same number of dimensions
        if lidar_features.dim() != cnn_features.dim():
            raise ValueError(f"Dimension mismatch: lidar_features {lidar_features.shape}, cnn_features {cnn_features.shape}")
        
        combined_features = th.cat([lidar_features, cnn_features], dim=-1)
        return combined_features

class CustomActor(nn.Module):
    def __init__(self, features_extractor, action_dim):
        super().__init__()
        self.features_extractor = features_extractor
        self.action_net = nn.Sequential(
            nn.Linear(256, 64),  # Use full feature vector
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, observations):
        features = self.features_extractor(observations)
        return self.action_net(features)

class CustomCritic(nn.Module):
    def __init__(self, features_extractor):
        super().__init__()
        self.features_extractor = features_extractor
        self.value_net = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, observations):
        features = self.features_extractor(observations)
        return self.value_net(features)

class LidarCritic(nn.Module):
    def __init__(self, features_extractor):
        super().__init__()
        self.features_extractor = features_extractor
        self.value_net = nn.Sequential(
            nn.Linear(128, 64),  # Only use first half of feature vector (lidar features)
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, observations):
        features = self.features_extractor(observations)
        return self.value_net(features[:, :128])  # Only use lidar features

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch as th
import torch.nn as nn

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[])
        self.features_extractor = CustomCombinedExtractor(self.observation_space)
        self.actor = CustomActor(self.features_extractor, self.action_space.shape[0])
        self.critic = CustomCritic(self.features_extractor)
        self.lidar_critic = LidarCritic(self.features_extractor)

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi = self.actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, None, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi = self.actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        value = self.critic(features)
        lidar_value = self.lidar_critic(features)
        return value, log_prob, distribution.entropy(), lidar_value

class MultiCriticPPO(PPO):
    def __init__(self, policy, env, **kwargs):
        # Wrap the environment with our custom wrapper
        env = TurtlebotEnvWrapper(env)
        
        super().__init__(policy, env, **kwargs)
        self.lidar_critic_loss = None
        self.critic_weight = 0.5  # Weight for combining critic outputs
        self.intrinsic_reward_coef = 0.01  # Coefficient for intrinsic reward

    def train(self):
        # Reset env
        obs = self.env.reset()
        # Initialize ICM
        icm = ICM(self.observation_space["camera"].shape, self.action_space.shape[0])
        icm_optimizer = th.optim.Adam(icm.parameters(), lr=1e-3)
        
        for iteration in range(self.n_iterations):
            print(f"Starting iteration {iteration}")
            
            # Collect rollouts
            for step in range(self.n_steps):
                actions, values, log_probs, lidar_values = self.policy.forward(obs)
                
                # Get intrinsic reward from ICM (only for camera data)
                with th.no_grad():
                    next_state_pred, action_pred = icm(obs["camera"], actions)
                    intrinsic_reward = F.mse_loss(next_state_pred, obs["camera"]).item()
                
                next_obs, rewards, dones, infos = self.env.step(actions)
                
                # Combine extrinsic and intrinsic rewards
                combined_rewards = rewards + self.intrinsic_reward_coef * intrinsic_reward
                
                # Store transition
                self.rollout_buffer.add(obs, actions, combined_rewards, dones, values, log_probs, lidar_values)
                
                obs = next_obs
                
            # Compute returns and advantages
            last_values, _, _, last_lidar_values = self.policy.forward(obs)
            self.rollout_buffer.compute_returns_and_advantage(last_values, dones)
            
            # Optimize policy and value function
            for epoch in range(self.n_epochs):
                print(f"  Epoch {epoch}")
                for rollout_data in self.rollout_buffer.get(self.batch_size):
                    actions = rollout_data.actions
                    observations = rollout_data.observations
                    
                    # ICM update
                    next_state_pred, action_pred = icm(observations["camera"], actions)
                    icm_loss = F.mse_loss(next_state_pred, observations["camera"]) + F.mse_loss(action_pred, actions)
                    icm_optimizer.zero_grad()
                    icm_loss.backward()
                    icm_optimizer.step()
                    
                    # PPO update
                    values, log_prob, entropy, lidar_values = self.policy.evaluate_actions(observations, actions)
                    
                    # Combine critic values
                    combined_values = self.critic_weight * values + (1 - self.critic_weight) * lidar_values
                    
                    advantages = rollout_data.advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    # PPO loss
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                    
                    value_loss = F.mse_loss(combined_values, rollout_data.returns)
                    
                    loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                    
                    # Gradient step
                    self.policy.optimizer.zero_grad()
                    loss.backward()
                    self.policy.optimizer.step()
                    
                    print(f"    Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")
            
            # Update learning rate
            self.lr_schedule.step()
            
        return self


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

class TurtlebotEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({
            "lidar": env.observation_space["lidar"],
            "camera": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        })

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        processed_obs = self._process_observation(observation)
        return processed_obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(observation)
        return processed_obs, reward, terminated, truncated, info

    def _process_observation(self, observation):
        lidar_data = np.array(observation["lidar"], dtype=np.float32)
        camera_data = observation["camera"]
        
        print(f"Original camera shape: {camera_data.shape}")
    
        if camera_data.shape == (3, 64, 64):
            camera_data = np.transpose(camera_data, (1, 2, 0))
        elif camera_data.shape == (64, 3, 64):
            camera_data = np.transpose(camera_data, (0, 2, 1))

        # Ensure camera data is (64, 64, 3)
        if camera_data.shape != (64, 64, 3):
            raise ValueError(f"Expected camera shape (64, 64, 3), got {camera_data.shape}")
        
        
        # Transpose and ensure correct shape
        camera_data = np.transpose(camera_data, (2, 0, 1))
        
        print(f"Transposed camera shape: {camera_data.shape}")
        
        processed_obs = {
            "lidar": lidar_data,
            "camera": camera_data.astype(np.float32) / 255.0  # Normalize to [0, 1]
        }
        
        # Print shapes for debugging
        print(f"Processed observation shapes: Lidar {processed_obs['lidar'].shape}, Camera {processed_obs['camera'].shape}")
        
        return processed_obs


    def _get_shapes(self, observation):
        return {key: value.shape for key, value in observation.items()}

# Usage
env = Turtlebot_Nav_Env()
wrapped_env = TurtlebotEnvWrapper(env)

callback = SaveOnBestTrainingRewardCallback(1000, "tmp/")

# Test the environment
print("Testing environment...")
obs, _ = wrapped_env.reset()
print(f"Reset observation shapes: {wrapped_env._get_shapes(obs)}")

action = wrapped_env.action_space.sample()
obs, reward, done, truncated, info = wrapped_env.step(action)
print(f"Step observation shapes: {wrapped_env._get_shapes(obs)}")

# Create the model
print("Creating model...")
model = MultiCriticPPO(CustomActorCriticPolicy, wrapped_env, verbose=1)

# Train the model
print("Starting training...")
model.learn(total_timesteps=10000)

# Save the trained model
model.save("multi_critic_ppo_model")
