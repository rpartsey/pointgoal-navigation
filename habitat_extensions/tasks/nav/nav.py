from typing import Any

from habitat import Config
from habitat.core.registry import registry
from habitat.tasks.nav.nav import TopDownMap as TopDownMapBase
from habitat.utils.visualizations import maps

import numpy as np

@registry.register_measure
class TopDownMap(TopDownMapBase):
    def __init__(
            self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        super().__init__(sim, config, *args, **kwargs)
        self.line_thickness = config.LINE_THICKNESS
        self.meters_per_pixel = config.METERS_PER_PIXEL
        # set FOG_OF_WAR.DRAW = False

    def _is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=2.0):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height < height + 0.1 < ref_floor_height + ceiling_height

    def get_original_map(self):
        top_down_map = maps.get_topdown_map_from_sim(
            self._sim,
            draw_border=self._config.DRAW_BORDER,
            meters_per_pixel=self.meters_per_pixel
        )

        self._fog_of_war_mask = None

        return top_down_map

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        # self._draw_goals_view_points(episode)
        # self._draw_goals_aabb(episode)
        # self._draw_goals_positions(episode)

        self._draw_shortest_path(episode, agent_position)

        # if self._config.DRAW_SOURCE:
        #     self._draw_point(
        #         episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
        #     )