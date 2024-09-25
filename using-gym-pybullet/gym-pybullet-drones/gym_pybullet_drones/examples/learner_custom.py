import os
import time
from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from Exploration_custom import ExplorationAviary

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_MAZE_SIZE = 5  # Size of the random maze
DEFAULT_GOAL_POSITION = np.array([4, 4, 1])  # The goal position in the maze

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, record_video=DEFAULT_RECORD_VIDEO, local=True):

    filename = os.path.join(output_folder, 'explore-save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Create the training and evaluation environments ####
    train_env = make_vec_env(ExplorationAviary,
                             env_kwargs=dict(
                                 maze_size=DEFAULT_MAZE_SIZE,
                                 goal_position=DEFAULT_GOAL_POSITION
                             ),
                             n_envs=1,
                             seed=0
                             )
    eval_env = ExplorationAviary(maze_size=DEFAULT_MAZE_SIZE, goal_position=DEFAULT_GOAL_POSITION)

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    #### Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                verbose=1)

    #### Reward threshold to stop training #####################
    target_reward = -1.0  # Adjust based on the maze complexity and goal
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward,
                                                     verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=1000,
                                 deterministic=True,
                                 render=False)

    #### Learn the policy ######################################
    model.learn(total_timesteps=int(1e6) if local else int(1e2),
                callback=eval_callback,
                log_interval=100)

    #### Save the final model ##################################
    model.save(filename+'/final_model.zip')
    print(f'Model saved to {filename}/final_model.zip')

    #### Print training progression ############################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(f"{data['timesteps'][j]},{data['results'][j][0]}")

    #### Evaluation ############################################
    if os.path.isfile(filename+'/best_model.zip'):
        model = PPO.load(filename+'/best_model.zip')
    else:
        print("[ERROR]: No model under the specified path")
        return

    #### Test the trained model ################################
    test_env = ExplorationAviary(gui=gui,
                                 maze_size=DEFAULT_MAZE_SIZE,
                                 goal_position=DEFAULT_GOAL_POSITION,
                                 record=record_video)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=1,
                    output_folder=output_folder)

    mean_reward, std_reward = evaluate_policy(model, test_env, n_eval_episodes=10)
    print(f"\n\n\nMean reward: {mean_reward} +- {std_reward}\n\n")

    obs, info = test_env.reset(seed=42)
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2) * test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)

        print(f"Obs: {obs} | Action: {action} | Reward: {reward} | Terminated: {terminated} | Truncated: {truncated}")

        logger.log(drone=0,
                   timestamp=i/test_env.CTRL_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], action.squeeze()]),
                   control=np.zeros(12))

        test_env.render()
        sync(i, start, test_env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            obs = test_env.reset(seed=42)

    test_env.close()
    logger.plot()

if __name__ == '__main__':
    run()
