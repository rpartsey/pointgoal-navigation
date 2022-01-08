import argparse
import copy
import cv2
import os
import random
from collections import OrderedDict
from typing import Optional, Union, Dict, Any

import numba
import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

import habitat
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.tasks.nav.nav import (
    PointGoalSensor, IntegratedPointGoalGPSAndCompassSensor,
    EpisodicGPSSensor, EpisodicCompassSensor, StopAction
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import batch_obs

from odometry.config.default import get_config as get_vo_config
from odometry.dataset import make_transforms
from odometry.models import make_model
from habitat_extensions.sensors.egomotion_pg_sensor import PointGoalEstimator


ACTION_INDEX_TO_NAME = {
    0: 'STOP',
    1: 'MOVE_FORWARD',
    2: 'TURN_LEFT',
    3: 'TURN_RIGHT'
}


def get_action_name(action_index: int):
    return ACTION_INDEX_TO_NAME[action_index]


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class PPOAgent(Agent):
    def __init__(self, config: Config, use_gps=False) -> None:
        self.use_gps = use_gps
        self.input_type = config.INPUT_TYPE
        try:
            self.obs_transforms = get_active_obs_transforms(config)
        except AttributeError:  # transform could not be imported
            print(
                "!!!WARNING!!! Transforms below were not imported and can't be used:\n",
                config.RL.POLICY.OBS_TRANSFORMS.ENABLED_TRANSFORMS
            )
            self.obs_transforms = []
        self.action_spaces = self._get_action_spaces(config)
        observation_spaces = self._get_observation_spaces(config)

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.RL.PPO.hidden_size

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        _seed_numba(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        policy = baseline_registry.get_policy(config.RL.POLICY.name)
        self.actor_critic = policy.from_config(
            config, observation_spaces, self.action_spaces
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def _get_observation_spaces(self, config):
        image_size = config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER
        if "ObjectNav" in config.TASK_CONFIG.TASK.TYPE:
            OBJECT_CATEGORIES_NUM = 20
            spaces = {
                ObjectGoalSensor.cls_uuid: Box(
                    low=0,
                    high=OBJECT_CATEGORIES_NUM,
                    shape=(1,),
                    dtype=np.int64
                ),
                EpisodicCompassSensor.cls_uuid: Box(
                    low=-np.pi,
                    high=np.pi,
                    shape=(1,),
                    dtype=np.float32
                ),
                EpisodicGPSSensor.cls_uuid: Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        else:
            if self.use_gps:
                sensor_uuid = IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            else:
                sensor_uuid = PointGoalSensor.cls_uuid
            spaces = {
                sensor_uuid: Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                )
            }

        if config.INPUT_TYPE in ["depth", "rgbd"]:
            spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(image_size.HEIGHT, image_size.WIDTH, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(image_size.HEIGHT, image_size.WIDTH, 3),
                dtype=np.uint8,
            )
        observation_spaces = SpaceDict(spaces)
        observation_spaces = apply_obs_transforms_obs_space(
            observation_spaces, self.obs_transforms
        )

        return observation_spaces

    def _get_action_spaces(self, config):
        return Discrete(6) if "ObjectNav" in config.TASK_CONFIG.TASK.TYPE else Discrete(4)

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, observations: Observations) -> Dict[str, int]:
        batch = batch_obs([observations], device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        with torch.no_grad():
            (_, actions, _, self.test_recurrent_hidden_states) = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        return {"action": actions[0][0].item()}


# TODO: come up with a more descriptive class name
class PPOAgentV2(PPOAgent):
    def __init__(self, config: Config, pointgoal_estimator: PointGoalEstimator):
        super().__init__(config)
        self.pointgoal_estimator = pointgoal_estimator

    def _get_observation_spaces(self, config):
        observation_spaces = super()._get_observation_spaces(config)
        observation_spaces.spaces.pop(PointGoalSensor.cls_uuid)
        observation_spaces.spaces[IntegratedPointGoalGPSAndCompassSensor.cls_uuid] = Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
        return observation_spaces

    def _get_pointgoal_estimate(self, observations):
        if get_action_name(self.prev_actions.item()) == StopAction.name:  # indicates the moment after the episode reset
            self.pointgoal_estimator.reset(observations)
            pointgoal = observations[PointGoalSensor.cls_uuid]
        else:
            pointgoal = self.pointgoal_estimator(observations, action=self.prev_actions.item())

        return pointgoal

    def act(self, observations: Observations) -> Dict[str, int]:
        # inject estimated point goal location as a 'pointgoal_with_gps_compass' sensor measure
        pointgoal = self._get_pointgoal_estimate(copy.deepcopy(observations))
        observations[IntegratedPointGoalGPSAndCompassSensor.cls_uuid] = pointgoal
        self.pred_rho_theta = pointgoal

        observations.pop(PointGoalSensor.cls_uuid)
        if self.input_type == 'depth':
            observations.pop('rgb')

        return super().act(observations)


class ShortestPathFollowerAgent(Agent):
    def __init__(self, env, goal_radius):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=goal_radius,
            return_one_hot=False
        )

    def act(self, observations) -> Union[int, str, Dict[str, Any]]:
        return self.shortest_path_follower.get_next_action(
            self.env.current_episode.goals[0].position
        )

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["PPOAgentV2", "PPOAgent", "ShortestPathFollowerAgent"],
        default="PPOAgentV2"
    )
    parser.add_argument("--input-type", type=str, choices=["rgb", "depth", "rgbd"], default="rgbd")
    parser.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    parser.add_argument("--ddppo-config-path", type=str, required=False)
    parser.add_argument("--ddppo-checkpoint-path", type=str, required=False)
    parser.add_argument("--vo-config-path", type=str, default="vo_config.yaml")
    parser.add_argument("--vo-checkpoint-path", type=str, default="vo.ckpt.pth")
    parser.add_argument("--rotation-regularization-on", action='store_true')
    parser.add_argument("--vertical-flip-on", action='store_true')
    parser.add_argument("--pth-gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config_paths = os.environ["CHALLENGE_CONFIG_FILE"]
    config = get_config(
        args.ddppo_config_path, ["BASE_TASK_CONFIG_PATH", config_paths]
    ).clone()
    config.defrost()
    config.RANDOM_SEED = args.seed
    config.PTH_GPU_ID = args.pth_gpu_id
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.ddppo_checkpoint_path
    config.freeze()

    if args.evaluation == "local":
        challenge = habitat.Challenge(eval_remote=False)
        challenge._env.seed(config.RANDOM_SEED)
    else:
        challenge = habitat.Challenge(eval_remote=True)

    if args.agent_type == PPOAgent.__name__:
        agent = PPOAgent(config)

    elif args.agent_type == PPOAgentV2.__name__:
        vo_config = get_vo_config(args.vo_config_path, new_keys_allowed=True)
        device = torch.device('cuda', args.pth_gpu_id)

        obs_transforms = make_transforms(vo_config.val.dataset.transforms)
        vo_model = make_model(vo_config.model).to(device)
        checkpoint = torch.load(args.vo_checkpoint_path, map_location=device)
        # if config.distrib_backend:
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            new_checkpoint[k.replace('module.', '')] = v
        checkpoint = new_checkpoint
        vo_model.load_state_dict(checkpoint)
        vo_model.eval()

        pointgoal_estimator = PointGoalEstimator(
            obs_transforms=obs_transforms,
            vo_model=vo_model,
            action_embedding_on=vo_config.model.params.action_embedding_size > 0,
            depth_discretization_on=(hasattr(vo_config.val.dataset.transforms, 'DiscretizeDepth')
                                     and vo_config.val.dataset.transforms.DiscretizeDepth.params.n_channels > 0),
            rotation_regularization_on=args.rotation_regularization_on,
            vertical_flip_on=args.vertical_flip_on,
            device=device
        )
        agent = PPOAgentV2(config, pointgoal_estimator)

    elif args.agent_type == ShortestPathFollowerAgent.__name__:
        assert args.evaluation == "local", "ShortestPathFollowerAgent supports only local evaluation"

        agent = ShortestPathFollowerAgent(
            env=challenge._env,
            goal_radius=config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE
        )

    else:
        raise ValueError(f'{args.agent_type} agent type doesn\'t exist!' )

    challenge.submit(agent)


if __name__ == "__main__":
    main()
