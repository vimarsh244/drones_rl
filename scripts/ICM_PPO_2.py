import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import os

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
            nn.Linear(256, action_dim)
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
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
    def __init__(self, state_dim, action_dim, hidden_dim=64, gru_hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        self.gru = nn.GRU(state_dim, gru_hidden_dim, batch_first=True)
        
        self.actor_mean = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(gru_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.gru_hidden_dim = gru_hidden_dim
        self.gru_hidden = None
    
    def forward(self, state, hidden=None):
        # Ensure state is 3D: (batch_size, sequence_length, input_size)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        
        if hidden is None:
            batch_size = state.size(0)
            hidden = self.init_hidden(batch_size)
        
        gru_out, hidden = self.gru(state, hidden)
        gru_out = gru_out[:, -1, :]  # Take the last output for each sequence
        
        action_mean = self.actor_mean(gru_out)
        action_std = self.actor_logstd.exp()
        value = self.critic(gru_out)
        
        return action_mean, action_std, value, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.gru_hidden_dim)

class PPO_ICM:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon_clip=0.2, epochs=10, icm_beta=0.5):
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
    
    def get_action(self, state, hidden):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean, action_std, _, new_hidden = self.actor_critic(state, hidden)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)  # Clip action to [-1, 1]
            return action.numpy().flatten(), dist.log_prob(action).sum(dim=-1), new_hidden
    
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
    
    def update(self, states, actions, old_log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        for _ in range(self.epochs):
            action_mean, action_std, values, _ = self.actor_critic(states)
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
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + self.icm_beta * icm_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, num_episodes, batch_size=64, save_interval=100):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            hidden = self.actor_critic.init_hidden(batch_size=1)
            
            states, actions, rewards, next_states, log_probs, values, dones = [], [], [], [], [], [], []
            
            while not (done or truncated):
                action, log_prob, new_hidden = self.get_action(state, hidden)
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                log_probs.append(log_prob)
                _, _, value, _ = self.actor_critic(torch.FloatTensor(state).unsqueeze(0), hidden)
                values.append(value.item())
                dones.append(done or truncated)
                
                state = next_state
                hidden = new_hidden
                episode_reward += reward
                
                if len(states) == batch_size:
                    # Compute returns and advantages
                    _, _, next_value, _ = self.actor_critic(torch.FloatTensor(next_state).unsqueeze(0), hidden)
                    returns, advantages = self.compute_gae(rewards, values, next_value.item(), dones)
                    
                    # Update the policy
                    self.update(states, actions, log_probs, returns, advantages)
                    
                    states, actions, rewards, next_states, log_probs, values, dones = [], [], [], [], [], [], []
                    hidden = self.actor_critic.init_hidden(batch_size=1)
            
            print(f"Episode {episode + 1}, Reward: {episode_reward}")
            
            # Save model at specified intervals
            if (episode + 1) % save_interval == 0:
                self.save_model(f"ppo_icm_gru_model_episode_{episode + 1}")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'icm_state_dict': self.icm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.icm.load_state_dict(checkpoint['icm_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

# Usage example
from turtlebot_nav_env import Turtlebot_Nav_Env

env = Turtlebot_Nav_Env()
agent = PPO_ICM(env)

# Train the agent
agent.train(num_episodes=2000, save_interval=100)

# Load a saved model
# agent.load_model("ppo_icm_gru_model_episode_100")

# Continue training or evaluate the loaded model
# agent.train(num_episodes=500, save_interval=100)