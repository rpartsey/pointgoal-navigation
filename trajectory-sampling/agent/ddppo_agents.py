#!/usr/bin/env python3

import argparse
import random

import numba
import numpy as np
import torch
from gym.spaces import Box, Dict, Discrete

import habitat
from habitat import Config
from habitat.core.agent import Agent
from habitat_baselines.config.default import get_config

from habitat_baselines.utils.common import batch_obs
from habitat_baselines.common.baseline_registry import baseline_registry
import habitat_extensions.sensors  # noqa - required to register a sensor to baseline_registry


@numba.njit
def _seed_numba(seed: int):
    random.seed(seed)
    np.random.seed(seed)


class DDPPOAgent(Agent):
    """
    Used following implementation as an example
    https://github.com/facebookresearch/habitat-challenge/blob/ddf1575532aecc4df2f4cd4c5db173b8eada3e1e/ddppo_agents.py#L37
    """
    def __init__(self, config: Config):
        if "ObjectNav" in config.TASK_CONFIG.TASK.TYPE:
            OBJECT_CATEGORIES_NUM = 20
            spaces = {
                "objectgoal": Box(
                    low=0, high=OBJECT_CATEGORIES_NUM, shape=(1,), dtype=np.int64
                ),
                "compass": Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float),
                "gps": Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        else:
            spaces = {
                "pointgoal": Box(
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
                shape=(
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH,
                    1,
                ),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT,
                    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH,
                    3,
                ),
                dtype=np.uint8,
            )
        observation_spaces = Dict(spaces)

        action_space = Discrete(len(config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS))

        self.device = torch.device("cuda:{}".format(config.TORCH_GPU_ID))
        self.hidden_size = config.RL.PPO.hidden_size

        random.seed(config.RANDOM_SEED)
        np.random.seed(config.RANDOM_SEED)
        _seed_numba(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True

        policy = baseline_registry.get_policy(config.RL.POLICY.name)
        self.actor_critic = policy.from_config(
            config, observation_spaces, action_space
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            print(f"Checkpoint loaded: {config.MODEL_PATH}")
            #  Filter only actor_critic weights
            self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                }
            )

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            1,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device, dtype=torch.bool)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)

    def act(self, observations):
        batch = batch_obs([observations], device=self.device)

        with torch.no_grad():
            _, action, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(action)

        return {'action': action.item()}


def main():
    """
    Smoke test if agent is instantiated properly and has expected navigation performance
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="../../config_files/challenge_pointnav2020_gt_loc.local.rgbd.yaml"
    )
    parser.add_argument(
        "--model-path",
        default="/home/rpartsey/code/3d-navigation/related_works/egolocalization/checkpoints/pointnav2020_gt_loc_rgbd_ckpt.199.pth"
    )
    args = parser.parse_args()

    config = get_config(
        "../../config_files/ddppo/ddppo_pointnav.yaml", ["BASE_TASK_CONFIG_PATH", args.config_file]
    ).clone()

    config.defrost()
    config.INPUT_TYPE = "rgbd"
    config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = 7
    config.freeze()

    agent = DDPPOAgent(config)
    challenge = habitat.Challenge(eval_remote=False)
    challenge._env.seed(config.RANDOM_SEED)
    challenge.submit(agent)


if __name__ == "__main__":
    main()
