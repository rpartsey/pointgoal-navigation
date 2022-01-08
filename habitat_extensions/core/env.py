from glob import glob

import gym
import numpy as np
from habitat.config import Config
from odometry.dataset.dataset import EgoMotionDatasetResized


class MockEpisodeNavEnv(gym.Env):
    action_to_id = {
        'STOP': 0,
        'FWD': 1,
        'LEFT': 2,
        'RIGHT': 3
    }

    id_to_action = {
        0: 'STOP',
        1: 'FWD',
        2: 'LEFT',
        3: 'RIGHT',
    }

    def __init__(self, config: Config, **kwargs):
        self._config = config
        self._cartesian_goal = None
        self._pointgoal = None
        self._actions = []
        self._observation_paths = []

    def reset(self):
        nav_observations_dir = self._config.ENVIRONMENT.OBS_DIR
        log_file_path = f'{nav_observations_dir}/{self._config.ENVIRONMENT.LOG_FILE_NAME}'

        # load goal, actions, and depth/rgb paths
        self._parse_cartesian_goal(log_file_path)
        self._parse_actions(log_file_path)
        self._load_observation_paths(nav_observations_dir)

        # reverse actions and depth/rgb paths
        self._actions.reverse()
        self._observation_paths.reverse()

        self._pointgoal = self._compute_pointgoal(np.zeros((3, )), self._cartesian_goal)
        action = self.action_to_id.get('EPISODE_RESET')  # always None
        depth_path, rgb_path = self._observation_paths.pop()

        observations = {
            'action': action,
            'pointgoal': self._pointgoal,
            'depth': EgoMotionDatasetResized.read_depth(depth_path),
            'rgb': EgoMotionDatasetResized.read_rgb(rgb_path),  # expects png to be stored as RGB (not BGR)
        }

        return observations

    def step(self, action):
        assert self.episode_over is False, "Episode over, call reset before calling step"

        action = self.action_to_id.get(self._actions.pop())
        depth_path, rgb_path = self._observation_paths.pop()

        observations = {
            'action': action,
            'pointgoal': self._pointgoal,
            'depth': EgoMotionDatasetResized.read_depth(depth_path),
            'rgb': EgoMotionDatasetResized.read_rgb(rgb_path),  # expects png to be stored as RGB (not BGR)
        }

        return observations

    def render(self, mode='human'):
        pass

    @property
    def episode_over(self) -> bool:
        return not bool(self._actions and self._observation_paths)

    def _compute_pointgoal(self, agent_state, goal):
        agent_x, agent_y, agent_rotation = agent_state
        agent_coordinates = np.array([agent_x, agent_y])
        rho = np.linalg.norm(agent_coordinates - goal)
        theta = (
            np.arctan2(goal[1] - agent_coordinates[1], goal[0] - agent_coordinates[0])
            - agent_rotation
        )
        theta = theta % (2 * np.pi)
        if theta >= np.pi:
            theta -= 2 * np.pi

        # return np.array([rho, theta], dtype=np.float32)

        return np.array([7.79685, 0.516399], dtype=np.float32)


    def _parse_cartesian_goal(self, log_file_path):
        with open(log_file_path, 'r') as log_file:
            for _ in range(2):
                line = log_file.readline()

        goal_str_list = line.replace('Goal location:', '').replace('[', '').replace(']', '').strip().split()
        self._cartesian_goal = np.array([float(i) for i in goal_str_list], dtype=np.float32)

    def _parse_actions(self, log_file_path):
        with open(log_file_path, 'r') as log_file:
            for _ in range(2):
                log_file.readline()

            for line in log_file:
                if 'STOP WAS CALLED' in line:
                    # self._actions.append(self.action_to_id['STOP'])
                    break

                action_name = line.split()[1]
                self._actions.append(action_name)

    def _load_observation_paths(self, nav_observations_dir):
        for file_path in sorted(glob(f'{nav_observations_dir}/*depth.png')):
            self._observation_paths.append((file_path, file_path.replace('depth', 'rgb')))
