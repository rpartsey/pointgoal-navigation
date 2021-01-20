from typing import Any

import quaternion
import numpy as np
import torch

from habitat_sim import utils
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.utils import cartesian_to_polar
from habitat.tasks.nav.nav import PointGoalSensor, NavigationEpisode

from odometry.config.default import get_config as get_train_config
from odometry.dataset import make_transforms
from odometry.models import make_model
from train_odometry import transform_batch


@registry.register_sensor
class PointGoalWithEgoPredictionsSensor(PointGoalSensor):
    """
    Sensor for PointGoal observations which are used in the PointNav task.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
    """
    cls_uuid: str = 'pointgoal_with_ego_predictions'

    def __init__(self, sim: Simulator, config: Config):
        self.pointgoal = None
        self.current_episode_id = None
        self.prev_agent_state = None
        self.prev_observations = None

        vo_model_train_config = get_train_config(config.TRAIN_CONFIG_PATH)
        self.device = torch.device('cuda', config.GPU_DEVICE_ID)
        self.observations_transforms = make_transforms(vo_model_train_config.train.dataset.transforms)
        self.vo_model = make_model(vo_model_train_config.model).to(self.device)
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=self.device)
        self.vo_model.load_state_dict(checkpoint['model_state'])
        self.vo_model.eval()

        super().__init__(sim=sim, config=config)

    def get_observation(self, observations, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        curr_agent_state = self._sim.get_agent_state()

        episode_id = (episode.episode_id, episode.scene_id)
        # beginning of episode
        if self.current_episode_id != episode_id:
            # init pointgoal using agent state + goal location
            self.current_episode_id = episode_id

            ref_position = curr_agent_state.position
            rotation_world_agent = curr_agent_state.rotation

            T_agent_to_scene = np.zeros(
                shape=(4, 4),
                dtype=np.float32
            )
            T_agent_to_scene[3, 3] = 1.
            T_agent_to_scene[0:3, 3] = ref_position
            T_agent_to_scene[0:3, 0:3] = (
                quaternion.as_rotation_matrix(rotation_world_agent)
            )

            T_scene_to_agent = np.linalg.inv(T_agent_to_scene)
            goal = np.array(episode.goals[0].position, dtype=np.float32)
            direction_vector_agent = np.dot(
                T_scene_to_agent,
                np.concatenate((goal, np.asarray([1.])), axis=0)
            )

            assert direction_vector_agent[3] == 1.
            self.pointgoal = direction_vector_agent[:3]
            self.prev_agent_state = curr_agent_state
            self.prev_observations = observations

            if self._goal_format == 'POLAR':
                rho, phi = cartesian_to_polar(
                    -direction_vector_agent[2], direction_vector_agent[0]
                )
                direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

            return direction_vector_agent

        # middle of episode
        # update pointgoal using egomotion
        visual_observations = {
            'source_rgb': self.prev_observations['rgb'],
            'target_rgb': observations['rgb'],
            'source_depth': self.prev_observations['depth'],
            'target_depth': observations['depth']
        }
        visual_observations = self.observations_transforms(visual_observations)
        batch = {k: v.unsqueeze(0) for k, v in visual_observations.items()}
        batch, _ = transform_batch(batch)
        batch = batch.to(self.device)
        with torch.no_grad():
            egomotion_preds = self.vo_model(batch)

        noisy_x, noisy_y, noisy_z, noisy_yaw = (
            egomotion_preds[0].item(),
            egomotion_preds[1].item(),
            egomotion_preds[2].item(),
            egomotion_preds[3].item()
        )

        # re-contruct the transformation matrix
        # using the noisy estimates for (x, y, z, yaw)
        noisy_translation = np.asarray([
            noisy_x,
            noisy_y,
            noisy_z,
        ], dtype=np.float32)
        noisy_rot_mat = quaternion.as_rotation_matrix(
            utils.quat_from_angle_axis(
                theta=noisy_yaw,
                axis=np.asarray([0, 1, 0])
            )
        )

        noisy_T_curr2prev_state = np.zeros(
            shape=(4, 4),
            dtype=np.float32
        )
        noisy_T_curr2prev_state[3, 3] = 1.
        noisy_T_curr2prev_state[0:3, 0:3] = noisy_rot_mat
        noisy_T_curr2prev_state[0:3, 3] = noisy_translation

        noisy_T_prev2curr_state = np.linalg.inv(
            noisy_T_curr2prev_state
        )

        direction_vector_agent = np.dot(
            noisy_T_prev2curr_state,
            np.concatenate(
                (self.pointgoal, np.asarray([1.], dtype=np.float32)),
                axis=0
            )
        )
        assert direction_vector_agent[3] == 1.
        self.pointgoal = direction_vector_agent[:3]
        self.prev_agent_state = curr_agent_state
        self.prev_observations = observations

        if self._goal_format == 'POLAR':
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent