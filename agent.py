import argparse
import copy
import gzip
import json
import os
import random
from collections import OrderedDict, defaultdict
from typing import Optional, Union, Dict, Any

import cv2
import numba
import numpy as np
import quaternion
import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

import habitat
from habitat import Benchmark as BaseBenchmark
from habitat.config import Config
from habitat.core.agent import Agent
from habitat.core.simulator import Observations
from habitat.tasks.nav.nav import (
    PointGoalSensor, IntegratedPointGoalGPSAndCompassSensor,
    EpisodicGPSSensor, EpisodicCompassSensor, StopAction
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.utils.visualizations import maps
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.common import batch_obs
from habitat.core.logging import logger
from tqdm import tqdm

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
    def __init__(self, config: Config) -> None:
        self.input_type = config.INPUT_TYPE
        self.obs_transforms = get_active_obs_transforms(config)
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
            spaces = {
                PointGoalSensor.cls_uuid: Box(
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

        observations.pop(PointGoalSensor.cls_uuid)
        if self.input_type == 'depth':
            observations.pop('rgb')

        return super().act(observations)


def get_polar_angle(ref_rotation):
    # agent_state = self._sim.get_agent_state()
    # quaternion is in x, y, z, w format
    # ref_rotation = agent_state.rotation

    heading_vector = quaternion_rotate_vector(
        ref_rotation.inverse(), np.array([0, 0, -1])
    )

    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    z_neg_z_flip = np.pi
    return np.array(phi) + z_neg_z_flip


def get_vo_agent_position(scene_agent_state, vo_pointgoal, scene_pointgoal):
    T_agent_to_scene = np.zeros((4, 4), dtype=np.float32)
    T_agent_to_scene[3, 3] = 1.
    T_agent_to_scene[:3, 3] = scene_agent_state.position
    T_agent_to_scene[:3, :3] = quaternion.as_rotation_matrix(scene_agent_state.rotation)

    gt_pointgoal = np.dot(
        np.linalg.inv(T_agent_to_scene),  # T_scene_to_agent
        np.concatenate((scene_pointgoal, np.asarray([1.])), axis=0)
    )[:3]

    vo_agent_position = scene_agent_state.position + (vo_pointgoal - gt_pointgoal)

    return vo_agent_position


class Benchmark(BaseBenchmark):
    def __init__(self, config_path, dest_tdm_trajectory_dir, eval_remote=False):
        self.dest_tdm_trajectory_dir = dest_tdm_trajectory_dir
        super().__init__(config_path, eval_remote=eval_remote)

    def submit(self, agent):
        metrics = super().evaluate(agent)
        for k, v in metrics.items():
            logger.info("{}: {}".format(k, v))

    def local_evaluate(
            self, agent: "PPOAgentV2", num_episodes: Optional[int] = None
    ) -> Dict[str, float]:
        if num_episodes is None:
            num_episodes = len(self._env.episodes)
        else:
            assert num_episodes <= len(self._env.episodes), (
                "num_episodes({}) is larger than number of episodes "
                "in environment ({})".format(
                    num_episodes, len(self._env.episodes)
                )
            )

        assert num_episodes > 0, "num_episodes should be greater than 0"

        topdown_map_dir = f'{self.dest_tdm_trajectory_dir}/topdown_map'
        json_dir = f'{self.dest_tdm_trajectory_dir}/json'

        os.makedirs(topdown_map_dir)
        os.makedirs(json_dir)

        env = self._env
        sim = self._env.sim

        agg_metrics: Dict = defaultdict(float)

        count_episodes = 0

        with tqdm(total=num_episodes) as pbar:
            while count_episodes < num_episodes:
                agent.reset()
                observations = env.reset()

                current_episode = env.current_episode
                agent_state = sim.get_agent_state()
                spf_points = [point.tolist() for point in sim.get_straight_shortest_path_points(
                    agent_state.position, current_episode.goals[0].position
                )]
                vo_agent_states = {
                    'position': [agent_state.position.tolist()],
                    'rotation': [get_polar_angle(agent_state.rotation)]
                }
                gt_agent_states = {
                    'position': [agent_state.position.tolist()],
                    'rotation': [get_polar_angle(agent_state.rotation)]
                }

                while not env.episode_over:
                    action = agent.act(observations)
                    observations = self._env.step(action)

                    vo_agent_egomotion = agent.pointgoal_estimator.egomotion
                    vo_agent_position = get_vo_agent_position(
                        agent_state, agent.pointgoal_estimator.pointgoal, current_episode.goals[0].position
                    )
                    vo_agent_states['position'].append(vo_agent_position.tolist())
                    vo_agent_states['rotation'].append(vo_agent_states['rotation'][-1]+vo_agent_egomotion[3])

                    agent_state = sim.get_agent_state()
                    gt_agent_states['position'].append(agent_state.position.tolist())
                    gt_agent_states['rotation'].append(get_polar_angle(agent_state.rotation))

                vo_agent_egomotion = agent.pointgoal_estimator.egomotion
                vo_agent_position = get_vo_agent_position(
                    agent_state, agent.pointgoal_estimator.pointgoal, current_episode.goals[0].position
                )
                vo_agent_states['position'].append(vo_agent_position.tolist())
                vo_agent_states['rotation'].append(vo_agent_states['rotation'][-1] + vo_agent_egomotion[3])

                vo_agent_states['position'] = vo_agent_states['position'][1:]
                vo_agent_states['rotation'] = vo_agent_states['rotation'][1:]

                metrics = env.get_metrics()
                metrics.update(current_episode.info)

                metric_strs = []
                for k in ('spl', 'success', 'softspl', 'distance_to_goal', 'geodesic_distance'):
                    metric_strs.append(f"{k}={metrics[k]:.2f}")

                top_down_map_config = env._config.TASK.TOP_DOWN_MAP
                top_down_map = maps.get_topdown_map_from_sim(
                    env.sim,
                    map_resolution=top_down_map_config.MAP_RESOLUTION,
                    draw_border=top_down_map_config.DRAW_BORDER,
                )

                spf_points = [
                    maps.to_grid(p[2], p[0], top_down_map.shape[0:2], sim=sim)
                    for p in spf_points
                ]
                vo_agent_states['position'] = [
                    maps.to_grid(p[2], p[0], top_down_map.shape[0:2], sim=sim)
                    for p in vo_agent_states['position']
                ]
                gt_agent_states['position'] = [
                    maps.to_grid(p[2], p[0], top_down_map.shape[0:2], sim=sim)
                    for p in gt_agent_states['position']
                ]

                scene_name = os.path.basename(current_episode.scene_id).split('.')[0]
                episode_id = current_episode.episode_id.zfill(3)
                top_down_map_name = f'{scene_name}_{episode_id}_' + '_'.join(metric_strs) + '.png'
                top_down_map_path = f'{topdown_map_dir}/{top_down_map_name}'

                cv2.imwrite(top_down_map_path, top_down_map)

                json_name = f'{scene_name}_{episode_id}.json.gz'
                json_path = f'{json_dir}/{json_name}'

                goal = current_episode.goals[0].position
                start = current_episode.start_position

                meta = {
                    'top_down_map_path': top_down_map_path,
                    'gt_agent_states': gt_agent_states,
                    'vo_agent_states': vo_agent_states,
                    'spf_points': spf_points,
                    'episode': {
                        'scene_name': scene_name,
                        'episode_id': episode_id,
                        'goal': maps.to_grid(goal[2], goal[0], top_down_map.shape[0:2], sim=sim),
                        'start': maps.to_grid(start[2], start[0], top_down_map.shape[0:2], sim=sim),
                        'metrics': metrics
                    }
                }

                with gzip.open(json_path, 'wt') as f:
                    json.dump(meta, f)

                for m, v in metrics.items():
                    agg_metrics[m] += v
                count_episodes += 1

                pbar.update()

        avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

        return avg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["PPOAgentV2", "PPOAgent", "ShortestPathFollowerAgent"],
        default="PPOAgentV2"
    )
    parser.add_argument("--input-type", type=str, choices=["rgb", "depth", "rgbd"], default="depth")
    parser.add_argument("--evaluation", type=str, default="local")
    parser.add_argument("--dest-dir", type=str, default="/home/rpartsey/code/3d-navigation/pointgoal-navigation/navtrajectory/val_mini")
    parser.add_argument("--challenge-config-path", type=str, default="config_files/challenge_pointnav2021.local.rgbd.yaml")
    parser.add_argument("--ddppo-config-path", type=str, required=False)
    parser.add_argument("--ddppo-checkpoint-path", type=str, required=False)
    parser.add_argument("--vo-config-path", type=str, default="vo_config.yaml")
    parser.add_argument("--vo-checkpoint-path", type=str, default="vo.ckpt.pth")
    parser.add_argument("--rotation-regularization-on", action='store_true')
    parser.add_argument("--vertical-flip-on", action='store_true')
    parser.add_argument("--pth-gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    config_path = args.challenge_config_path
    config = get_config(
        args.ddppo_config_path, ["BASE_TASK_CONFIG_PATH", config_path]
    ).clone()
    config.defrost()
    config.RANDOM_SEED = args.seed
    config.PTH_GPU_ID = args.pth_gpu_id
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.ddppo_checkpoint_path
    config.freeze()

    challenge = Benchmark(config_path, args.dest_dir, eval_remote=False)
    challenge._env.seed(config.RANDOM_SEED)

    vo_config = get_vo_config(args.vo_config_path, new_keys_allowed=True)
    device = torch.device('cuda', args.pth_gpu_id)

    obs_transforms = make_transforms(vo_config.val.dataset.transforms)
    vo_model = make_model(vo_config.model).to(device)
    checkpoint = torch.load(args.vo_checkpoint_path, map_location=device)

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

    challenge.submit(agent)


if __name__ == "__main__":
    main()
