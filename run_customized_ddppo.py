import habitat_extensions.sensors  # noqa - required to register new sensors to baseline_registry
import habitat_baselines_extensions  # noqa - required to register custom habitat baselines extensions
import habitat_baselines_extensions.rl.ddppo.policy  # noqa - required to register custom PointNavResNetPolicy
from habitat_baselines.run import main


if __name__ == "__main__":
    main()
