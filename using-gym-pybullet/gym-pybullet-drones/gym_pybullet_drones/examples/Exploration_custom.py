import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pubullet_drones. import rayTestBatch
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ExplorationAviary(BaseRLAviary):
    """Single agent RL exploration problem: navigate through a maze using LiDAR."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 maze_size=5,
                 goal_position=None
                 ):
        """Initialization of a single agent RL exploration environment.

        Parameters are similar to the hover task, but we introduce a goal position
        and maze size.

        """
        self.TARGET_POS = goal_position if goal_position is not None else np.array([maze_size-1, maze_size-1, 1])
        self.MAZE_SIZE = maze_size
        self.EPISODE_LEN_SEC = 15  # Increased time limit for exploration
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """Reward is based on progress toward the target goal."""
        state = self._getDroneStateVector(0)
        dist_to_goal = np.linalg.norm(self.TARGET_POS - state[0:3])
        reward = -dist_to_goal  # Penalize distance to the goal
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Terminates when the goal is reached."""
        state = self._getDroneStateVector(0)
        if np.linalg.norm(self.TARGET_POS - state[0:3]) < 0.1:
            return True  # Success if drone is within 0.1 meters of the target
        return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Truncate if the drone moves out of bounds or time runs out."""
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > self.MAZE_SIZE or abs(state[1]) > self.MAZE_SIZE or state[2] > 2.0  # Out of bounds
            or abs(state[7]) > 0.4 or abs(state[8]) > 0.4):  # Too much tilt
            return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:  # Time-out
            return True
        return False

    ################################################################################
    
    def _getObservation(self):
        """Fetches the drone's state and LiDAR data."""
        state = self._getDroneStateVector(0)
        lidar_data = self._getLidarData()
        return np.concatenate([state, lidar_data])

    ################################################################################
    
    def _getLidarData(self):
        """Simulate LiDAR data around the drone."""
        lidar_data = rayTestBatch()
        return lidar_data

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s)."""
        return {"info": "Exploration task"}

