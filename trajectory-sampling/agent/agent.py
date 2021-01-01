import argparse
import random

import numpy as np
import torch
from gym.spaces import Discrete, Dict, Box

from . import Policy
from . import batch_obs

import habitat
from habitat import Config
from habitat.core.agent import Agent

import ipdb

from .ddppo_agents import DDPPOAgent


def get_defaut_config():
    c = Config()
    c.INPUT_TYPE = "depth"
    c.MODEL_PATH = "checkpoints/depth.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.NUM_RECURRENT_LAYERS = 2
    c.RNN_TYPE = "LSTM"
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    return c


class PPOAgent(Agent):
    def __init__(self, config: Config):
        self.config = config
        if config.GOAL_SENSOR_UUID == "pointgoal":
            spaces = {
                config.GOAL_SENSOR_UUID: Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(3,),
                    dtype=np.float32,
                )
            }
        else:
            spaces = {
                config.GOAL_SENSOR_UUID: Box(
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
                shape=(config.RESOLUTION, config.RESOLUTION, 1),
                dtype=np.float32,
            )

        if config.INPUT_TYPE in ["rgb", "rgbd"]:
            spaces["rgb"] = Box(
                low=0,
                high=255,
                shape=(config.RESOLUTION, config.RESOLUTION, 3),
                dtype=np.uint8,
            )
        observation_spaces = Dict(spaces)

        action_spaces = Discrete(4)

        self.device = (
            torch.device("cuda:{}".format(config.PTH_GPU_ID))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.hidden_size = config.HIDDEN_SIZE

        random.seed(config.RANDOM_SEED)
        torch.random.manual_seed(config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True

        self.actor_critic = Policy(
            observation_space=observation_spaces,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            num_recurrent_layers=config.NUM_RECURRENT_LAYERS,
            rnn_type=config.RNN_TYPE,
            goal_sensor_uuid=config.GOAL_SENSOR_UUID
        )
        self.actor_critic.to(self.device)

        if config.MODEL_PATH:
            ckpt = torch.load(config.MODEL_PATH, map_location=self.device)
            #  Filter only actor_critic weights
            incompatible_keys = self.actor_critic.load_state_dict(
                {
                    k.replace("actor_critic.", ""): v
                    for k, v in ckpt["state_dict"].items()
                    if "actor_critic" in k
                },
                strict=False
            )
            if (
                len(incompatible_keys.missing_keys) > 0 or
                len(incompatible_keys.unexpected_keys) > 0
            ):
                assert len(incompatible_keys.missing_keys) == 2
                assert "net.critic_linear.weight" in incompatible_keys.missing_keys
                assert "net.critic_linear.bias" in incompatible_keys.missing_keys

                assert len(incompatible_keys.unexpected_keys) == 2
                assert "critic.fc.weight" in incompatible_keys.unexpected_keys
                assert "critic.fc.bias" in incompatible_keys.unexpected_keys

        else:
            habitat.logger.error(
                "Model checkpoint wasn't loaded, evaluating " "a random model."
            )

        self.test_recurrent_hidden_states = None
        self.not_done_masks = None
        self.prev_actions = None

    def reset(self):
        self.test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers, 1, self.hidden_size, device=self.device
        )
        self.not_done_masks = torch.zeros(1, 1, device=self.device)
        self.prev_actions = torch.zeros(1, 1, device=self.device, dtype=torch.long)

    def act(self, observations):
        batch = batch_obs([observations])

        for sensor in batch:
            batch[sensor] = batch[sensor].to(self.device)

        if self.config.GOAL_SENSOR_UUID == "pointgoal":
            pg = batch[self.config.GOAL_SENSOR_UUID]
            batch["pointgoal"] = torch.stack(
                [pg[..., 0], torch.cos(-pg[..., 1]), torch.sin(-pg[..., 1])], dim=-1
            )

        with torch.no_grad():
            _, actions, _, self.test_recurrent_hidden_states = self.actor_critic.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks = torch.ones(1, 1, device=self.device)
            self.prev_actions = actions.clone()

        return actions[0][0].item()


def build_agent(config):
    return DDPPOAgent(config)