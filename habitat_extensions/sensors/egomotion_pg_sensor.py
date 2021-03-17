from typing import Any

import quaternion
import numpy as np
import torch

import habitat_sim
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
    INVERSE_ACTION = {
        2: 3,
        3: 2
    }

    def __init__(self, sim: Simulator, config: Config, dataset=None, task=None):
        # vo init
        vo_config = get_train_config(config.TRAIN_CONFIG_PATH, new_keys_allowed=True)
        self.device = torch.device('cuda', config.GPU_DEVICE_ID)
        self.obs_transforms = make_transforms(vo_config.val.dataset.transforms)
        self.vo_model = make_model(vo_config.model).to(self.device)
        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=self.device)
        self.vo_model.load_state_dict(checkpoint)
        self.vo_model.eval()

        # sensor init
        self.pointgoal = None
        self.prev_observations = None
        self.action_embedding_on = vo_config.model.params.action_embedding_size > 0
        self.collision_embedding_on = vo_config.model.params.collision_embedding_size > 0
        self.depth_discretization_on = vo_config.val.dataset.transforms.DiscretizeDepth.params.n_channels > 0
        self.rotation_regularization_on = config.ROTATION_REGULARIZATION

        super().__init__(sim=sim, config=config)

    def _compote_direction_vector(self, observations, episode, **kwargs):
        episode_reset = 'action' not in kwargs
        episode_end = (not episode_reset) and (kwargs['action']['action'] == 0)

        if episode_end:
            return np.concatenate((self.pointgoal, np.asarray([1.], dtype=np.float32)), axis=0)

        elif episode_reset:
            agent_state = self._sim.get_agent_state()

            T_agent_to_scene = np.zeros((4, 4), dtype=np.float32)
            T_agent_to_scene[3, 3] = 1.
            T_agent_to_scene[:3, 3] = agent_state.position
            T_agent_to_scene[:3, :3] = (quaternion.as_rotation_matrix(agent_state.rotation))

            goal = np.array(episode.goals[0].position, dtype=np.float32)

            return np.dot(
                np.linalg.inv(T_agent_to_scene),  # T_scene_to_agent
                np.concatenate((goal, np.asarray([1.])), axis=0)
            )

        else:
            # update pointgoal using egomotion
            x, y, z, yaw = self._compute_egomotion(observations, **kwargs)

            # recontruct the transformation matrix using the noisy estimates for (x, y, z, yaw)
            noisy_translation = np.asarray([x, y, z], dtype=np.float32)
            noisy_rot_mat = quaternion.as_rotation_matrix(
                habitat_sim.utils.quat_from_angle_axis(theta=yaw, axis=np.asarray([0, 1, 0]))
            )

            noisy_T_curr2prev_state = np.zeros((4, 4), dtype=np.float32)
            noisy_T_curr2prev_state[3, 3] = 1.
            noisy_T_curr2prev_state[:3, 3] = noisy_translation
            noisy_T_curr2prev_state[:3, :3] = noisy_rot_mat

            return np.dot(
                np.linalg.inv(noisy_T_curr2prev_state),  # noisy_T_prev2curr_state
                np.concatenate((self.pointgoal, np.asarray([1.], dtype=np.float32)), axis=0)
            )

    def _compute_egomotion(self, observations, **kwargs):
        visual_obs = {
            'source_rgb': self.prev_observations['rgb'],
            'target_rgb': observations['rgb'],
            'source_depth': self.prev_observations['depth'],
            'target_depth': observations['depth']
        }
        if self.action_embedding_on:
            visual_obs.update({'action': kwargs['action']['action'] - 1})  # shift action ids by 1 as we don't use STOP
        if self.collision_embedding_on:
            visual_obs.update({'collision': int(self._sim.previous_step_collided)})

        visual_obs = self.obs_transforms(visual_obs)
        batch = {k: v.unsqueeze(0) for k, v in visual_obs.items()}

        if self.rotation_regularization_on and kwargs['action']['action'] in self.ROTATION_ACTIONS:
            batch.update({
                'source_rgb': torch.cat([batch['source_rgb'], batch['target_rgb']], 0),
                'target_rgb': torch.cat([batch['target_rgb'], batch['source_rgb']], 0),
                'source_depth': torch.cat([batch['source_depth'], batch['target_depth']], 0),
                'target_depth': torch.cat([batch['target_depth'], batch['source_depth']], 0),
            })
            if self.depth_discretization_on:
                batch.update({
                    'source_depth_discretized': torch.cat([batch['source_depth_discretized'], batch['target_depth_discretized']], 0),
                    'target_depth_discretized': torch.cat([batch['target_depth_discretized'], batch['source_depth_discretized']], 0)
                })
            if self.action_embedding_on:
                action = batch['action']
                inverse_action = action.clone().fill_(self.INVERSE_ACTION[kwargs['action']['action']]-1)
                batch.update({
                    'action': torch.cat([action, inverse_action], 0)
                })
            if self.collision_embedding_on:
                batch.update({
                    'collision': torch.cat([batch['collision'], batch['collision'].clone()], 0)
                })

        batch, embeddings, _ = transform_batch(batch)
        batch = batch.to(self.device)
        for k, v in embeddings.items():
            embeddings[k] = v.to(self.device)

        with torch.no_grad():
            egomotion = self.vo_model(batch, **embeddings)
            if egomotion.size(0) == 2:
                egomotion = (egomotion[:1] + -egomotion[1:]) / 2

        return egomotion.squeeze(0).cpu()

    def get_observation(self, observations, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        direction_vector_agent = self._compote_direction_vector(observations, episode, **kwargs)
        assert direction_vector_agent[3] == 1.

        self.pointgoal = direction_vector_agent[:3]
        self.prev_observations = observations

        if self._goal_format == 'POLAR':
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent
