import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import gym
import numpy as np
from torch.distributions import Normal

# Assuming you have your Turtlebot_Nav_Env in a file named turtlebot_env.py
from turtlebot_nav_env import Turtlebot_Nav_Env

# Hyperparameters
GAMMA = 0.99
LAMBDA = 1.0
ENTROPY_BETA = 0.01
LEARNING_RATE = 1e-4
MAX_EPISODES = 10000
MAX_STEPS = 3000
NUM_PROCESSES = 1
UPDATE_INTERVAL = 5
ICM_BETA = 0.2

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.actor_mean = nn.Linear(256, num_outputs)
        self.actor_std = nn.Parameter(torch.zeros(num_outputs))
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_mean = torch.tanh(self.actor_mean(x))
        action_std = F.softplus(self.actor_std)
        value = self.critic(x)
        return action_mean, action_std, value

class ICM(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(ICM, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.inverse = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_size + num_outputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, state, next_state, action):
        state_feat = self.feature(state)
        next_state_feat = self.feature(next_state)
        pred_action = self.inverse(torch.cat([state_feat, next_state_feat], 1))
        pred_next_state_feat = self.forward_model(torch.cat([state_feat, action], 1))
        return pred_action, pred_next_state_feat, next_state_feat

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                                         weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()
                state['step'].share_memory_()

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def worker(rank, shared_model, shared_icm, optimizer, counter, lock):
    torch.manual_seed(rank)
    env = Turtlebot_Nav_Env()
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    icm = ICM(env.observation_space.shape[0], env.action_space.shape[0])

    for episode in range(MAX_EPISODES):
        model.load_state_dict(shared_model.state_dict())
        icm.load_state_dict(shared_icm.state_dict())
        
        state, _ = env.reset()
        done = False
        truncated = False
        values = []
        log_probs = []
        rewards = []
        entropies = []
        intrinsic_rewards = []

        for step in range(MAX_STEPS):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, action_std, value = model(state_tensor)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            log_prob = dist.log_prob(action).sum()
            entropy = dist.entropy().sum()

            next_state, reward, done, truncated, _ = env.step(action.squeeze().numpy())

            # Compute intrinsic reward
            pred_action, pred_next_state_feat, next_state_feat = icm(state_tensor, torch.FloatTensor(next_state).unsqueeze(0), action)
            intrinsic_reward = F.mse_loss(pred_next_state_feat, next_state_feat.detach()).item()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            intrinsic_rewards.append(intrinsic_reward)

            state = next_state

            if done or truncated or (step + 1) % UPDATE_INTERVAL == 0:
                if done or truncated:
                    R = 0
                else:
                    _, _, R = model(torch.FloatTensor(state).unsqueeze(0))
                    R = R.detach()

                values.append(R)
                policy_loss = 0
                value_loss = 0
                gae = 0
                for i in reversed(range(len(rewards))):
                    R = GAMMA * R + rewards[i] + ICM_BETA * intrinsic_rewards[i]
                    advantage = R - values[i]
                    value_loss += 0.5 * advantage.pow(2)
                    gae = gae * GAMMA * LAMBDA + advantage
                    policy_loss -= log_probs[i] * gae.detach() + ENTROPY_BETA * entropies[i]

                # ICM loss
                state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
                next_state_tensor = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
                action_tensor = torch.FloatTensor(np.array(action)).unsqueeze(0)
                pred_action, pred_next_state_feat, next_state_feat = icm(state_tensor, next_state_tensor, action_tensor)
                inverse_loss = F.mse_loss(pred_action, action_tensor)
                forward_loss = F.mse_loss(pred_next_state_feat, next_state_feat.detach())
                icm_loss = inverse_loss + forward_loss

                optimizer.zero_grad()
                (policy_loss + 0.5 * value_loss + icm_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 40)
                torch.nn.utils.clip_grad_norm_(icm.parameters(), 40)
                ensure_shared_grads(model, shared_model)
                ensure_shared_grads(icm, shared_icm)
                optimizer.step()

                with lock:
                    counter.value += 1

                values = []
                log_probs = []
                rewards = []
                entropies = []
                intrinsic_rewards = []

            if done or truncated:
                print(f"Process {rank}, Episode {episode}, Steps: {step}, Reward: {sum(rewards)}")
                break

if __name__ == '__main__':
    torch.manual_seed(42)
    env = Turtlebot_Nav_Env()
    shared_model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    shared_model.share_memory()
    shared_icm = ICM(env.observation_space.shape[0], env.action_space.shape[0])
    shared_icm.share_memory()

    optimizer = SharedAdam(list(shared_model.parameters()) + list(shared_icm.parameters()), lr=LEARNING_RATE)
    optimizer.share_memory()

    processes = []
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    for rank in range(NUM_PROCESSES):
        p = mp.Process(target=worker, args=(rank, shared_model, shared_icm, optimizer, counter, lock))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()