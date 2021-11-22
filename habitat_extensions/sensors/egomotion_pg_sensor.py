import copy
from collections import OrderedDict
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
from habitat.utils.geometry_utils import quaternion_from_coeff, quaternion_rotate_vector

from odometry.config.default import get_config as get_train_config
from odometry.dataset import make_transforms
from odometry.models import make_model
from odometry.utils import transform_batch
from odometry.utils.utils import polar_to_cartesian


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


class PointGoalEstimator:
    def __init__(
            self,
            obs_transforms,
            vo_model,
            action_embedding_on,
            depth_discretization_on,
            rotation_regularization_on,
            vertical_flip_on,
            device
    ):
        self.obs_transforms = obs_transforms
        self.vo_model = vo_model
        self.action_embedding_on = action_embedding_on
        self.depth_discretization_on = depth_discretization_on
        self.rotation_regularization_on = rotation_regularization_on
        self.vertical_flip_on = vertical_flip_on
        self.prev_observations = None
        self.pointgoal = None
        self.egomotion = np.array([0, 0, 0, 0], dtype=np.float32)
        self.device = device

    def _compute_pointgoal(self, x, y, z, yaw):
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

    def __call__(self, observations, action):
        visual_obs = {
            'source_rgb': self.prev_observations['rgb'],
            'target_rgb': observations['rgb'],
            'source_depth': self.prev_observations['depth'],
            'target_depth': observations['depth']
        }
        egomotion_estimates = self._compute_egomotion(visual_obs, action)

        if self.vertical_flip_on:
            vflip_visual_obs = {
                'source_rgb': np.fliplr(self.prev_observations['rgb']).copy(),
                'target_rgb': np.fliplr(observations['rgb']).copy(),
                'source_depth': np.fliplr(self.prev_observations['depth']).copy(),
                'target_depth': np.fliplr(observations['depth']).copy()
            }
            vflip_action = INVERSE_ACTION[action] if action in ROTATION_ACTIONS else action
            vflip_egomotion_estimates = self._compute_egomotion(vflip_visual_obs, vflip_action)

            egomotion_estimates = (egomotion_estimates + vflip_egomotion_estimates * torch.tensor([-1, 1, 1, -1])) / 2

        egomotion_estimates = egomotion_estimates.cpu().numpy()
        self.egomotion = egomotion_estimates

        direction_vector_agent_cart = self._compute_pointgoal(*egomotion_estimates)
        assert direction_vector_agent_cart[3] == 1.

        self.pointgoal = direction_vector_agent_cart[:3]
        self.prev_observations = observations

        rho, phi = cartesian_to_polar(-direction_vector_agent_cart[2], direction_vector_agent_cart[0])
        direction_vector_agent_polar = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent_polar

    def reset(self, observations):
        self.prev_observations = observations
        self.pointgoal = polar_to_cartesian(*observations[PointGoalSensor.cls_uuid])
        self.egomotion = np.array([0, 0, 0, 0], dtype=np.float32)

    def _compute_egomotion(self, visual_obs, action):
        if self.action_embedding_on:
            visual_obs['action'] = action - 1  # shift all action ids as we don't use 0 - STOP

        visual_obs = self.obs_transforms(visual_obs)
        batch = {k: v.unsqueeze(0) for k, v in visual_obs.items()}

        if self.rotation_regularization_on and (action in ROTATION_ACTIONS):
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
                batch_action = batch['action']
                inverse_action = batch_action.clone().fill_(INVERSE_ACTION[action] - 1)
                batch.update({
                    'action': torch.cat([batch_action, inverse_action], 0)
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


@registry.register_sensor
class EgomotionPointGoalSensor(PointGoalSensor):
    cls_uuid = 'pointgoal_with_gps_compass'

    def __init__(self, sim: Simulator, config: Config, dataset=None, task=None):
        device = torch.device('cuda', sim.habitat_config.HABITAT_SIM_V0.GPU_DEVICE_ID)

        vo_config = get_train_config(config.TRAIN_CONFIG_PATH, new_keys_allowed=True)
        vo_obs_transforms = make_transforms(vo_config.val.dataset.transforms)
        vo_model = make_model(vo_config.model).to(device)

        checkpoint = torch.load(config.CHECKPOINT_PATH, map_location=device)
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            new_checkpoint[k.replace('module.', '')] = v
        checkpoint = new_checkpoint
        vo_model.load_state_dict(checkpoint)
        vo_model.eval()

        action_embedding_on = vo_config.model.params.action_embedding_size > 0
        # collision_embedding_on = vo_config.model.params.collision_embedding_size > 0
        flip_on = config.FLIP_ON
        swap_on = config.SWAP_ON
        depth_discretization_on = (
                hasattr(vo_config.val.dataset.transforms, 'DiscretizeDepth')
                and vo_config.val.dataset.transforms.DiscretizeDepth.params.n_channels > 0
        )

        self.pointgoal_estimator = PointGoalEstimator(
            obs_transforms=vo_obs_transforms,
            vo_model=vo_model,
            action_embedding_on=action_embedding_on,
            depth_discretization_on=depth_discretization_on,
            rotation_regularization_on=swap_on,
            vertical_flip_on=flip_on,
            device=device
        )

        super().__init__(sim=sim, config=config)

    def _compote_direction_vector(self, observations, episode, **kwargs):
        episode_reset = 'action' not in kwargs
        episode_end = (not episode_reset) and (kwargs['action']['action'] == 0)

        if episode_reset:
            # at episode reset compute pointgoal and reset pointgoal_estimator
            source_position = np.array(episode.start_position, dtype=np.float32)
            source_rotation = quaternion_from_coeff(episode.start_rotation)
            goal_position = np.array(episode.goals[0].position, dtype=np.float32)

            direction_vector = goal_position - source_position
            direction_vector_agent = quaternion_rotate_vector(
                source_rotation.inverse(), direction_vector
            )
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent_polar = np.array([rho, -phi], dtype=np.float32)

            self.pointgoal_estimator.reset({**observations, PointGoalSensor.cls_uuid: direction_vector_agent_polar})

        elif not episode_end:
            self.pointgoal_estimator(observations, kwargs['action']['action'])
        else:
            pass

        return np.concatenate((self.pointgoal_estimator.pointgoal, np.asarray([1.], dtype=np.float32)), axis=0)

    def get_observation(self, observations, episode: NavigationEpisode, *args: Any, **kwargs: Any):
        direction_vector_agent = self._compote_direction_vector(observations, episode, **kwargs)
        assert direction_vector_agent[3] == 1.

        if self._goal_format == 'POLAR':
            rho, phi = cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            direction_vector_agent = np.array([rho, -phi], dtype=np.float32)

        return direction_vector_agent
