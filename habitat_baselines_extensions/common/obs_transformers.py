import copy
from typing import Dict, Tuple

import torch
from gym import spaces

from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer


@baseline_registry.register_obs_transformer()
class Duplicator(ObservationTransformer):
    def __init__(
            self,
            prefix: str,
            trans_keys: Tuple[str] = ("rgb", "depth")
    ):
        super().__init__()
        self._prefix: str = prefix
        self.trans_keys: Tuple[str] = trans_keys

    def prefix(self, key):
        return f"{self._prefix}_{key}"

    def transform_observation_space(
            self,
            observation_space: spaces.Dict,
            **kwargs
    ):
        observation_space_copy = copy.deepcopy(observation_space)
        for sensor_key in observation_space.spaces:
            if sensor_key in self.trans_keys:
                observation_space_copy.spaces[self.prefix(sensor_key)] = copy.deepcopy(
                    observation_space.spaces[sensor_key]
                )

        return observation_space_copy

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        observations.update(
            {
                self.prefix(sensor_key): copy.deepcopy(observations[sensor_key])
                for sensor_key in self.trans_keys
                if sensor_key in observations
            }
        )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.DUPLICATOR.OBS_KEY_PREFIX)
