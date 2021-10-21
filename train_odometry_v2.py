import argparse

from odometry.config.default import get_config
from odometry.trainers import make_trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        required=True,
        type=str,
        help='path to the configuration file'
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    config_path = args.config_file
    config = get_config(config_path, new_keys_allowed=True)

    trainer = make_trainer(config)
    trainer.train()
