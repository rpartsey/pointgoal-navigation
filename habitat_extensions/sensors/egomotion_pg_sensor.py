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
from odometry.utils import transform_batch


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

    # Note: cls_uuid is inherited from PointGoalSensor to assure compatibility with habitat-lab code
    ROTATION_ACTIONS = {
        # 0   STOP
        # 1   MOVE_FORWARD
        2,  # TURN_LEFT
        3,  # TURN_RIGHT
    }

    def __init__(self, sim: Simulator, config: Config, dataset=None, task=None):
        self.pointgoal = None
        self.current_episode_id = None
        self.prev_agent_state = None
        self.prev_observations = None
        self.rotation_regularization_on = config.ROTATION_REGULARIZATION

        vo_model_train_config = get_train_config(config.TRAIN_CONFIG_PATH, new_keys_allowed=True)
        self.device = torch.device('cuda', config.GPU_DEVICE_ID)
        self.obs_transforms = make_transforms(vo_model_train_config.train.dataset.transforms)
        self.vo_model = make_model(vo_model_train_config.model).to(self.device)
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=self.device)
        self.vo_model.load_state_dict(checkpoint)
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
        visual_obs = {
            'source_rgb': self.prev_observations['rgb'],
            'target_rgb': observations['rgb'],
            'source_depth': self.prev_observations['depth'],
            'target_depth': observations['depth']
        }
        visual_obs = self.obs_transforms(visual_obs)
        batch = {k: v.unsqueeze(0) for k, v in visual_obs.items()}

        if self.rotation_regularization_on and kwargs['action']['action'] in self.ROTATION_ACTIONS:
            batch.update({
                'source_rgb': torch.cat([batch['source_rgb'], batch['target_rgb']], 0),
                'target_rgb': torch.cat([batch['target_rgb'], batch['source_rgb']], 0),
                'source_depth': torch.cat([batch['source_depth'], batch['target_depth']], 0),
                'target_depth': torch.cat([batch['target_depth'], batch['source_depth']], 0),
            })
            if all(key in batch for key in ['source_depth_discretized', 'target_depth_discretized']):
                batch.update({
                    'source_depth_discretized': torch.cat([batch['source_depth_discretized'], batch['target_depth_discretized']], 0),
                    'target_depth_discretized': torch.cat([batch['target_depth_discretized'], batch['source_depth_discretized']], 0)
                })

        batch, _ = transform_batch(batch)
        batch = batch.to(self.device)
        with torch.no_grad():
            egomotion_preds = self.vo_model(batch)

        if egomotion_preds.size(0) == 2:
            noisy_x, noisy_y, noisy_z, noisy_yaw = ((egomotion_preds[0] + -egomotion_preds[1]) / 2).cpu()
        else:
            noisy_x, noisy_y, noisy_z, noisy_yaw = egomotion_preds.squeeze(0).cpu()

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