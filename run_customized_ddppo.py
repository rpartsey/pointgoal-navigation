import habitat_extensions.sensors  # noqa - required to register new sensors to baseline_registry
import habitat_extensions.tasks.nav.nav # noqa - required to register TopDownMap
import habitat_baselines_extensions  # noqa - required to register custom habitat baselines extensions
import habitat_baselines_extensions.rl.ddppo.policy  # noqa - required to register custom PointNavResNetPolicy
import habitat_baselines_extensions.rl.ppo.ppo_trainer  # noqa - required to register custom PPOTrainer
from habitat_baselines.run import main


if __name__ == "__main__":
    main()
