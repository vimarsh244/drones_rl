import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os
import csv

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, feature_dim=128):
        super(ICM, self).__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )
        
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax()
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )
    
    def forward(self, state, next_state, action):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)
        
        # Inverse model
        inverse_input = torch.cat([state_feat, next_state_feat], dim=1)
        pred_action = self.inverse_model(inverse_input)
        
        # Forward model
        forward_input = torch.cat([state_feat, action], dim=1)
        pred_next_state_feat = self.forward_model(forward_input)
        
        return pred_action, pred_next_state_feat, next_state_feat

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.tan_act = nn.Tanh()
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, state):
        action_mean = self.tan_act(self.actor_mean(state))
        action_std = self.actor_logstd.exp()
        return action_mean, action_std, self.critic(state)

class PPO_ICM:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon_clip=0.2, epochs=10, icm_beta=0.5, icm_scale=0.1, log_path='training_log.csv'):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim)
        self.icm = ICM(self.state_dim, self.action_dim)
        
        self.optimizer = optim.Adam(list(self.actor_critic.parameters()) + list(self.icm.parameters()), lr=learning_rate)
        
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.icm_beta = icm_beta
        self.icm_scale = icm_scale
        self.log_path = log_path
        
        # Initialize CSV logging
        with open(self.log_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Episode', 'ICM Inverse Loss', 'ICM Forward Loss', 'Actor Loss', 'Critic Loss', 'Total Loss', 'Rewards', 'Intrinsic Reward'])
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, _ = self.actor_critic(state)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)  # Clip action to [-1, 1]
            return action.numpy().flatten(), dist.log_prob(action).sum(dim=-1)
    
    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []
        last_advantage = 0
        last_value = next_value

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * last_value * (1 - dones[t]) - values[t]
            last_advantage = delta + gamma * lam * (1 - dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
            last_value = values[t]

        returns = np.array(advantages) + values
        advantages = np.array(advantages)
                
        return returns, advantages
    
    def update(self, states, actions, old_log_probs, returns, advantages, intrinsic_rewards):
        icm_inverse_loss = 0
        icm_forward_loss = 0
        actor_loss_value = 0
        critic_loss_value = 0
        total_loss_value = 0

        for _ in range(self.epochs):
            action_mean, action_std, values = self.actor_critic(states)
            dist = Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            
            values = values.squeeze()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            
            # ICM update
            next_states = torch.roll(states, -1, dims=0)
            pred_actions, pred_next_state_feats, next_state_feats = self.icm(states, next_states, actions)
            
            inverse_loss = nn.MSELoss()(pred_actions, actions)
            forward_loss = nn.MSELoss()(pred_next_state_feats, next_state_feats.detach())
            
            icm_loss = inverse_loss + forward_loss
            
            # Total loss
            loss = actor_loss + 0.005 * critic_loss - 0.01 * entropy + self.icm_beta * icm_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate losses for logging
            icm_inverse_loss += inverse_loss.item()
            icm_forward_loss += forward_loss.item()
            actor_loss_value += actor_loss.item()
            critic_loss_value += critic_loss.item()
            total_loss_value += loss.item()
        
        return icm_inverse_loss / self.epochs, icm_forward_loss / self.epochs, actor_loss_value / self.epochs, critic_loss_value / self.epochs, total_loss_value / self.epochs
    
    def train(self, num_episodes, batch_size=64, save_interval=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            cumulative_intrinsic_reward = 0

            states, actions, rewards, next_states, log_probs, values, dones, intrinsic_rewards = [], [], [], [], [], [], [], []
            
            while not (done or truncated):
                action, log_prob = self.get_action(state)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                log_probs.append(log_prob)
                values.append(self.actor_critic(torch.FloatTensor(state).unsqueeze(0))[-1].item())
                dones.append(done or truncated)
                
                # ICM intrinsic reward
                with torch.no_grad():
                    _, pred_next_state_feats, next_state_feats = self.icm(torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(next_state).unsqueeze(0), torch.FloatTensor(action).unsqueeze(0))
                    intrinsic_reward = self.icm_scale * ((pred_next_state_feats - next_state_feats).pow(2).sum().item())
                    intrinsic_rewards.append(intrinsic_reward)
                    cumulative_intrinsic_reward += intrinsic_reward
                
                state = next_state
                episode_reward += reward + intrinsic_reward
                
                if len(states) == batch_size:
                    states = torch.FloatTensor(np.array(states))
                    actions = torch.FloatTensor(np.array(actions))
                    old_log_probs = torch.FloatTensor(log_probs)
                    
                    # Compute returns and advantages
                    next_value = self.actor_critic(torch.FloatTensor(next_state).unsqueeze(0))[-1].item()
                    combined_rewards = [r + ir for r, ir in zip(rewards, intrinsic_rewards)]
                    returns, advantages = self.compute_gae(combined_rewards, values, next_value, dones)
                    returns = torch.FloatTensor(returns)
                    advantages = torch.FloatTensor(advantages)
                    
                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    
                    icm_inv_loss, icm_fwd_loss, act_loss, crit_loss, tot_loss = self.update(states, actions, old_log_probs, returns, advantages, intrinsic_rewards)
                    
                    states, actions, rewards, next_states, log_probs, values, dones, intrinsic_rewards = [], [], [], [], [], [], [], []
            
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
            
            # Log episode results
            with open(self.log_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([episode + 1, icm_inv_loss, icm_fwd_loss, act_loss, crit_loss, tot_loss, episode_reward, cumulative_intrinsic_reward])
            
            # Save model at specified intervals
            if (episode + 1) % save_interval == 0:
                self.save_model(f"/home/vimarsh/Desktop/ceeri/ws_rl/src/multi_critic_rl/scripts/output/ppo_icm_model_episode_{episode + 1}")

    def save_model(self, path):
        """Save the model to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'icm_state_dict': self.icm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load the model from the specified path."""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.icm.load_state_dict(checkpoint['icm_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")


from turtlebot_nav_env import Turtlebot_Nav_Env

# Usage example
env = Turtlebot_Nav_Env()

agent = PPO_ICM(env)

# Train the agent
agent.train(num_episodes=2000, save_interval=50)

# Load a saved model
# agent.load_model("/home/vimarsh/Desktop/ceeri/ws_rl/src/multi_critic_rl/scripts/output/ppo_icm_model_episode_1200")

# Continue training or evaluate the loaded model
# agent.train(num_episodes=800, save_interval=50)
