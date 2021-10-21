from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import NavRLEnv


class NavRLEnvExtension(NavRLEnv):
    """
    Extends habitat_baselines.common.environments.NavRLEnv by redefining get_reward method.

    Wraps terminal reward into a separate method.
    Doesn't change get_reward, just decouples get_reward into two methods.
    """

    def get_terminal_reward(self):
        return self._rl_config.SUCCESS_REWARD

    def get_reward(self, observations):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_success():
            reward += self.get_terminal_reward()

        return reward


@baseline_registry.register_env(name="NavRLEnvSPLReward")
class NavRLEnvSPLReward(NavRLEnvExtension):
    def get_terminal_reward(self):
        return 2.5 * self._env.get_metrics()["spl"]
