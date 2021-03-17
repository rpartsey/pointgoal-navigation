from typing import Union, Dict, Any

from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


class ShortestPathFollowerAgent(Agent):
    def __init__(self, env, goal_radius):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=env.habitat_env.sim,
            goal_radius=goal_radius,
            return_one_hot=False
        )

    def act(self, observations) -> Union[int, str, Dict[str, Any]]:
        return self.shortest_path_follower.get_next_action(
            self.env.habitat_env.current_episode.goals[0].position
        )

    def reset(self) -> None:
        pass
