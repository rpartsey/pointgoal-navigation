import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions as SimulatorActions

class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        self._previous_action = action
        return super().step(action)

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        if (
            self._previous_action == SimulatorActions.STOP
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info = {}

        if self.get_done(observations):
            info["spl"] = self.habitat_env.get_metrics()["spl"]
        return info