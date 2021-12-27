import copy
from typing import Dict, Tuple

import torch
from torch import Tensor
from gym import spaces

from habitat import Config
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.utils.common import get_image_height_width, overwrite_gym_box_shape


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


@baseline_registry.register_obs_transformer(name="Resize")
@baseline_registry.register_obs_transformer(name="ResizeDepth")
class ResizeDepth(ObservationTransformer):
    r"""An nn module the resizes depth input.
    This module assumes that all images in the batch are of the same size.
    """

    def __init__(
        self,
        size: int,
        channels_last: bool = True,
        trans_keys: Tuple[str] = ("depth", ),
    ):
        """Args:
        size: The size you want to resize the shortest edge to
        channels_last: indicates if channels is the last dimension
        """
        super(ResizeDepth, self).__init__()
        self._size: int = size
        self.channels_last: bool = channels_last
        self.trans_keys: Tuple[str] = trans_keys

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        size = self._size
        observation_space = copy.deepcopy(observation_space)
        if size:
            for key in observation_space.spaces:
                if key in self.trans_keys:
                    # In the observation space dict, the channels are always last
                    h, w = get_image_height_width(
                        observation_space.spaces[key], channels_last=True
                    )
                    if size == h == w:
                        continue
                    new_size = (size, size)
                    logger.info(
                        "Resizing observation of %s: from %s to %s"
                        % (key, (h, w), new_size)
                    )
                    observation_space.spaces[key] = overwrite_gym_box_shape(
                        observation_space.spaces[key], new_size
                    )
        return observation_space

    def _transform_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return self.image_resize(
            obs, self._size, channels_last=self.channels_last
        )

    @staticmethod
    def image_resize(img: Tensor, size: int, channels_last: bool = False):
        """
            Used habitat-lab image_resize_shortest_edge as an example.
            https://github.com/facebookresearch/habitat-lab/blob/786a5eec68cf3b4cf7134af615394c981d365a89/habitat_baselines/utils/common.py#L406
        """
        img = torch.as_tensor(img)
        no_batch_dim = len(img.shape) == 3
        if len(img.shape) < 3 or len(img.shape) > 5:
            raise NotImplementedError()
        if no_batch_dim:
            img = img.unsqueeze(0)  # Adds a batch dimension
        if channels_last:
            if len(img.shape) == 4:
                # NHWC -> NCHW
                img = img.permute(0, 3, 1, 2)
            else:
                # NDHWC -> NDCHW
                img = img.permute(0, 1, 4, 2, 3)

        img = torch.nn.functional.interpolate(
            img.float(), size=(size, size), mode="bilinear"
        ).to(dtype=img.dtype)

        if channels_last:
            if len(img.shape) == 4:
                # NCHW -> NHWC
                img = img.permute(0, 2, 3, 1)
            else:
                # NDCHW -> NDHWC
                img = img.permute(0, 1, 3, 4, 2)
        if no_batch_dim:
            img = img.squeeze(dim=0)  # Removes the batch dimension
        return img

    @torch.no_grad()
    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if self._size is not None:
            observations.update(
                {
                    sensor: self._transform_obs(observations[sensor])
                    for sensor in self.trans_keys
                    if sensor in observations
                }
            )
        return observations

    @classmethod
    def from_config(cls, config: Config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.RESIZE.SIZE)
