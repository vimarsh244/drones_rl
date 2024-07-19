import argparse
import logging
import os
import random
import time
from turtlebot_nav_env import Turtlebot_Nav_Env

import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_memory import ReplayMemory, Transition
from wrappers.normalized_actions import NormalizedActions

# Global variables (replacing argparse)
ENV_NAME = "Turtlebot_Nav_Env"
RENDER_TRAIN = False
RENDER_EVAL = False
LOAD_MODEL = False
SAVE_DIR = "./saved_models/"
SEED = 0
TIMESTEPS = int(1e6)
BATCH_SIZE = 32
REPLAY_SIZE = int(1e6)
GAMMA = 0.99
TAU = 0.001
NOISE_STDDEV = 0.2
HIDDEN_SIZE = (400, 300)
N_TEST_CYCLES = 10
NUM_EP = 1e4


# Set up logger
logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

# Libdom raises an error if this is not set to true on Mac OSX
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using {}".format(device))

if __name__ == "__main__":
    # Define the directory where to save and load models
    checkpoint_dir = SAVE_DIR + ENV_NAME
    writer = SummaryWriter('runs/run_1')

    env = Turtlebot_Nav_Env()

    # Define the reward threshold when the task is solved
    # reward_threshold = gym.spec(ENV_NAME).reward_threshold if gym.spec(ENV_NAME).reward_threshold is not None else np.inf
    reward_threshold = np.inf

    # Set random seed
    # env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define and build DDPG agent
    hidden_size = HIDDEN_SIZE
    agent = DDPG(GAMMA,
                 TAU,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    # Initialize replay memory
    memory = ReplayMemory(int(REPLAY_SIZE))

    # Initialize OU-Noise
    nb_actions = env.action_space.shape[-1]
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions),
                                            sigma=float(NOISE_STDDEV) * np.ones(nb_actions))

    # Initialize counters and variables
    start_step = 0
    timestep = start_step // 10000 + 1
    rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
    epoch = 0
    t = 0
    time_last_checkpoint = time.time()

    # Start training
    # logger.info('Train agent on {} env'.format({env.unwrapped.spec.id}))
    logger.info('Doing {} timesteps'.format(TIMESTEPS))
    logger.info('Start at timestep {0} with t = {1}'.format(timestep, t))
    logger.info('Start training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

    ep = 1

    while ep <= NUM_EP:
        ou_noise.reset()
        epoch_return = 0

        state = torch.Tensor([env.reset()]).to(device)
        env.controller.status_pub.publish(f"Episode : {ep}")
        timestep = 0
        while True:
            action = agent.calc_action(state, ou_noise)
            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
            timestep += 1
            epoch_return += reward

            mask = torch.Tensor([done]).to(device)
            reward = torch.Tensor([reward]).to(device)
            next_state = torch.Tensor([next_state]).to(device)

            memory.push(state, action, mask, next_state, reward)

            state = next_state

            epoch_value_loss = 0
            epoch_policy_loss = 0

            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_params(batch)

                epoch_value_loss += value_loss
                epoch_policy_loss += policy_loss

            if done:
                break

        rewards.append(epoch_return)
        value_losses.append(epoch_value_loss)
        policy_losses.append(epoch_policy_loss)
        writer.add_scalar('epoch/return', epoch_return, epoch)

        # Test every 10th episode
        if ep % 100 == 0:
            t += 1
            test_rewards = []
            for _ in range(N_TEST_CYCLES):
                state = torch.Tensor([env.reset()]).to(device)
                test_reward = 0
                while True:
                    if RENDER_EVAL:
                        env.render()

                    action = agent.calc_action(state)  # Selection without noise

                    next_state, reward, done, _ = env.step(action.cpu().numpy()[0])
                    test_reward += reward

                    next_state = torch.Tensor([next_state]).to(device)

                    state = next_state
                    if done:
                        break
                test_rewards.append(test_reward)

            mean_test_rewards.append(np.mean(test_rewards))

            for name, param in agent.actor.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            for name, param in agent.critic.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

            writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
            logger.info("Epoch: {}, current timestep: {}, last reward: {}, "
                        "mean reward: {}, mean test reward {}".format(epoch,
                                                                      ep,
                                                                      rewards[-1],
                                                                      np.mean(rewards[-10:]),
                                                                      np.mean(test_rewards)))

            # Save if mean of last three averaged rewards while testing is greater than reward threshold
            if np.mean(mean_test_rewards[-3:]) >= reward_threshold:
                agent.save_checkpoint(timestep, memory)
                time_last_checkpoint = time.time()
                logger.info('Saved model at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        epoch += 1

    agent.save_checkpoint(timestep, memory)
    logger.info('Saved model at endtime {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    logger.info('Stopping training at {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
    env.close()