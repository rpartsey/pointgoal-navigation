from typing import Any

import quaternion
import numpy as np
from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes

from odometry.dataset.utils import get_relative_egomotion


@registry.register_sensor
class EgomotionSensor(Sensor):
    cls_uuid: str = 'egomotion'

    def __init__(self, sim, config, **kwargs: Any):
        super().__init__(config=config)
        self._sim = sim
        self.prev_agent_state = None
        self.current_episode_id = None

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(4,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        episode_id = (episode.episode_id, episode.scene_id)
        agent_state = self._sim.get_agent_state()

        if self.current_episode_id != episode_id:
            self.current_episode_id = episode_id
            self.prev_agent_state = agent_state

        egomotion = get_relative_egomotion({
            'source_agent_state': {
                'position': self.prev_agent_state.position.tolist(),
                'rotation': quaternion.as_float_array(self.prev_agent_state.rotation).tolist()
            },
            'target_agent_state': {
                'position': agent_state.position.tolist(),
                'rotation': quaternion.as_float_array(agent_state.rotation).tolist()
            },
        })
        self.prev_agent_state = agent_state

        translation = egomotion['translation']
        rotation = np.expand_dims(egomotion['rotation'], 0)

        return np.concatenate((translation, rotation))
