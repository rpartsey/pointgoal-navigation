from typing import List, Optional, Union

from habitat_extensions.config.default import CN, CONFIG_FILE_SEPARATOR


# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.INPUT_TYPE = "depth"
_C.MODEL_PATH = "checkpoints/depth.pth"
_C.RESOLUTION = 256
_C.HIDDEN_SIZE = 512
_C.NUM_RECURRENT_LAYERS = 2
_C.RNN_TYPE = "LSTM"
_C.RANDOM_SEED = 7
_C.PTH_GPU_ID = 1 
_C.GOAL_SENSOR_UUID = "pointgoal"

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
