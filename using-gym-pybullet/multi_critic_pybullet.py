import gymnasium as gym
import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
import torch
import torch.nn as nn
import torch.nn.functional as F

class MazeDroneEnv(TakeoffAviary):
    def __init__(self):
        super().__init__(drone_model=DroneModel.CF2X,
                         initial_xyzs=np.array([[0., 0., 0.5]]),
                         initial_rpys=np.array([[0., 0., 0.]]),
                         physics=Physics.PYB,
                         freq=240,
                         aggregate_phy_steps=1,
                         gui=True,
                         record=False)
        
        # Create maze walls
        self.create_maze()
        
        # Define observation and action spaces
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(low=0, high=10, shape=(16,), dtype=np.float32),
            "camera": gym.spaces.Box(low=0, high=255, shape=(32, 32, 3), dtype=np.uint8)
        })
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    
    def create_maze(self):
        # Create walls for a simple maze
        wall_positions = [
            [2, 2, 1], [-2, -2, 1], [2, -2, 1], [-2, 2, 1],
            [0, 3, 1], [0, -3, 1], [3, 0, 1], [-3, 0, 1]
        ]
        for pos in wall_positions:
            p.loadURDF("cube.urdf", pos, globalScaling=0.5)
    
    def _computeObs(self):
        # Get drone state
        state = self._getDroneStateVector(0)
        
        # Simulate LiDAR
        lidar_data = self.simulate_lidar(state[:3])
        
        # Simulate camera
        camera_data = self.simulate_camera(state[:3], state[6:9])
        
        return {
            "lidar": lidar_data,
            "camera": camera_data
        }
    
    def simulate_lidar(self, position):
        lidar_data = []
        for i in range(16):
            angle = 2 * np.pi * i / 16
            direction = [np.cos(angle), np.sin(angle), 0]
            hit = p.rayTest(position, [position[0] + direction[0] * 10,
                                       position[1] + direction[1] * 10,
                                       position[2] + direction[2] * 10])[0]
            lidar_data.append(hit[2])
        return np.array(lidar_data)
    
    def simulate_camera(self, position, orientation):
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.1, farVal=100)
        view_matrix = p.computeViewMatrix(position, [position[0] + orientation[0],
                                                     position[1] + orientation[1],
                                                     position[2] + orientation[2]], [0, 0, 1])
        _, _, rgb_img, _, _ = p.getCameraImage(32, 32, view_matrix, proj_matrix)
        return rgb_img[:,:,:3]
    
    def step(self, action):
        # Convert normalized actions to RPM
        rpm = np.clip((action + 1) * 0.5, 0, 1) * self.MAX_RPM
        
        # Step the simulation
        obs, reward, done, info = super().step(rpm)
        
        # Compute custom reward
        pos = self._getDroneStateVector(0)[:3]
        reward = -np.linalg.norm(pos[:2])  # Negative distance from center
        
        # Check for collision
        if self.check_collision(pos):
            reward -= 10
            done = True
        
        return self._computeObs(), reward, done, False, info
    
    def check_collision(self, position):
        for i in range(p.getNumBodies()):
            if i == self.DRONE_IDS[0]:
                continue
            closest_points = p.getClosestPoints(bodyA=self.DRONE_IDS[0], bodyB=i, distance=0.1)
            if len(closest_points) > 0:
                return True
        return False

# Multi-Critic Network
class MultiCriticNetwork(nn.Module):
    def __init__(self, lidar_dim, camera_dim, action_dim):
        super(MultiCriticNetwork, self).__init__()
        
        # Lidar critic
        self.lidar_critic = nn.Sequential(
            nn.Linear(lidar_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Camera critic
        self.camera_critic = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7 + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, lidar, camera, action):
        lidar_value = self.lidar_critic(torch.cat([lidar, action], dim=1))
        camera_value = self.camera_critic(torch.cat([camera.view(-1, 3, 32, 32), action.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 32, 32)], dim=1))
        return lidar_value, camera_value

# Custom TD3 for multi-critic
class MultiCriticTD3(TD3):
    def __init__(self, policy, env, learning_rate=1e-3, buffer_size=1000000, learning_starts=100, batch_size=100, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, action_noise=None, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, tensorboard_log=None, create_eval_env=False, policy_kwargs=None, verbose=0, seed=None, device='auto', _init_setup_model=True):
        super().__init__(policy, env, learning_rate, buffer_size, learning_starts, batch_size, tau, gamma, train_freq, gradient_steps, action_noise, policy_delay, target_policy_noise, target_noise_clip, tensorboard_log, create_eval_env, policy_kwargs, verbose, seed, device, _init_setup_model)
        
        # Replace critics with multi-critic network
        self.critic = MultiCriticNetwork(16, 32*32*3, self.action_space.shape[0])
        self.critic_target = MultiCriticNetwork(16, 32*32*3, self.action_space.shape[0])
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
    
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Custom training loop for multi-critic
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            with torch.no_grad():
                next_actions = self.actor_target(replay_data.next_observations)
                noise = torch.randn_like(next_actions) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (next_actions + noise).clamp(-1, 1)
                
                lidar_next_values, camera_next_values = self.critic_target(replay_data.next_observations["lidar"], 
                                                                           replay_data.next_observations["camera"], 
                                                                           next_actions)
                next_values = torch.min(lidar_next_values, camera_next_values)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_values
            
            lidar_values, camera_values = self.critic(replay_data.observations["lidar"], 
                                                      replay_data.observations["camera"], 
                                                      replay_data.actions)
            critic_loss = F.mse_loss(lidar_values, target_q_values) + F.mse_loss(camera_values, target_q_values)
            
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            
            if self._n_updates % self.policy_delay == 0:
                actor_actions = self.actor(replay_data.observations)
                lidar_actor_values, camera_actor_values = self.critic(replay_data.observations["lidar"], 
                                                                      replay_data.observations["camera"], 
                                                                      actor_actions)
                actor_loss = -torch.min(lidar_actor_values, camera_actor_values).mean()
                
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            
            self._n_updates += 1

# Training script
env = MazeDroneEnv()

# Initialize the agent
model = MultiCriticTD3("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("multi_critic_drone")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()