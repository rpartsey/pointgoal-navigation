from typing import List, Optional, Union

import numpy as np
from habitat_extensions.config.default import CN, CONFIG_FILE_SEPARATOR, _get_default_config


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# BASELINE
# -----------------------------------------------------------------------------
_C.BASELINE = CN()
# -----------------------------------------------------------------------------
# REINFORCEMENT LEARNING (RL)
# -----------------------------------------------------------------------------
_C.BASELINE.RL = CN()
_C.BASELINE.RL.SUCCESS_REWARD = 10.0
_C.BASELINE.RL.SLACK_REWARD = -0.01
# -----------------------------------------------------------------------------
# ORBSLAM2 BASELINE
# -----------------------------------------------------------------------------
_C.BASELINE.ORBSLAM2 = CN()
_C.BASELINE.ORBSLAM2.SLAM_VOCAB_PATH = "baselines/slambased/data/ORBvoc.txt"
_C.BASELINE.ORBSLAM2.SLAM_SETTINGS_PATH = (
    "baselines/slambased/data/mp3d3_small1k.yaml"
)
_C.BASELINE.ORBSLAM2.MAP_CELL_SIZE = 0.1
_C.BASELINE.ORBSLAM2.MAP_SIZE = 40
_C.BASELINE.ORBSLAM2.CAMERA_HEIGHT = _get_default_config().SIMULATOR.DEPTH_SENSOR.POSITION[
    1
]
_C.BASELINE.ORBSLAM2.BETA = 100
_C.BASELINE.ORBSLAM2.H_OBSTACLE_MIN = 0.3 * _C.BASELINE.ORBSLAM2.CAMERA_HEIGHT
_C.BASELINE.ORBSLAM2.H_OBSTACLE_MAX = 1.0 * _C.BASELINE.ORBSLAM2.CAMERA_HEIGHT
_C.BASELINE.ORBSLAM2.D_OBSTACLE_MIN = 0.1
_C.BASELINE.ORBSLAM2.D_OBSTACLE_MAX = 4.0
_C.BASELINE.ORBSLAM2.PREPROCESS_MAP = True
_C.BASELINE.ORBSLAM2.MIN_PTS_IN_OBSTACLE = (
    _get_default_config().SIMULATOR.DEPTH_SENSOR.WIDTH / 2.0
)
_C.BASELINE.ORBSLAM2.ANGLE_TH = float(np.deg2rad(15))
_C.BASELINE.ORBSLAM2.DIST_REACHED_TH = 0.15
_C.BASELINE.ORBSLAM2.NEXT_WAYPOINT_TH = 0.5
_C.BASELINE.ORBSLAM2.NUM_ACTIONS = 3
_C.BASELINE.ORBSLAM2.DIST_TO_STOP = 0.05
_C.BASELINE.ORBSLAM2.PLANNER_MAX_STEPS = 500
_C.BASELINE.ORBSLAM2.DEPTH_DENORM = (
    _get_default_config().SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
)

# -----------------------------------------------------------------------------


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
