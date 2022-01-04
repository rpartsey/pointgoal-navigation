from habitat.core.registry import registry
from habitat.tasks.nav.nav import TopDownMap as TopDownMapBase


@registry.register_measure
class TopDownMap(TopDownMapBase):
    def _is_on_same_floor(self, height, ref_floor_height=None, ceiling_height=2.0):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height < height + 0.1 < ref_floor_height + ceiling_height