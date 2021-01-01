import sys
import csv
import ipdb
import random

from habitat_baselines.common.baseline_registry import baseline_registry

from . import NavRLEnv
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def get_gibson_scenes(config):
    # get all scenes
    all_gibson_scenes = PointNavDatasetV1.get_scenes_to_load(
        config.DATASET
    )

    # filter by votes (only high-quality envs)
    filtered_gibson_scenes = filter_gibson_scenes_by_votes(
        all_gibson_scenes,
        config.DATASET.GIBSON_VOTES_CSV
    )
    assert len(filtered_gibson_scenes) > 0
    return filtered_gibson_scenes

def filter_gibson_scenes_by_votes(
    all_scenes,
    votes_csv_file,
    votes_thresh=4.0):

    with open(votes_csv_file, 'r') as f:
        csv_file = csv.reader(f)
        _ = next(csv_file)
        gibson_scene_to_votes_map = {
            row[0]: float(row[1])
            for row in csv_file
        }

    filtered_gibson_scenes = [
        (scene, gibson_scene_to_votes_map[scene])
        for scene in all_scenes
        if gibson_scene_to_votes_map[scene] >= votes_thresh
    ]

    return filtered_gibson_scenes

def update_config_with_scene_names(env_config, scene_names):
    if isinstance(scene_names, str):
        scene_names = [scene_names]

    env_config.defrost()
    env_config.DATASET.CONTENT_SCENES = scene_names
    env_config.freeze()

    return env_config

def get_pointnav_episodes_for_random_scene(
    config,
    filtered_gibson_scenes):
    
    # randomly sample a scene
    random.shuffle(filtered_gibson_scenes)
    scene, _ = filtered_gibson_scenes[0]

    config = update_config_with_scene_names(config, scene)
    dataset = PointNavDatasetV1(config.DATASET)

    return config, dataset

def get_pointnav_episodes_for_all_scenes(
    config,
    filtered_gibson_scenes):

    random.shuffle(filtered_gibson_scenes)
    scene_names = [item[0] for item in filtered_gibson_scenes]

    config = update_config_with_scene_names(config, scene_names)
    dataset = PointNavDatasetV1(config.DATASET)

    return config, dataset

def build_env(config):
    random.seed(config.RANDOM_SEED)  # TODO: check if this line can be removed

    task_config = config.TASK_CONFIG
    filtered_gibson_scenes = get_gibson_scenes(task_config)
    if task_config.DATASET.SINGLE_SCENE_TEST:
        env_config, dataset = get_pointnav_episodes_for_random_scene(
            task_config,
            filtered_gibson_scenes
        )
    else:
        env_config, dataset = get_pointnav_episodes_for_all_scenes(
            task_config,
            filtered_gibson_scenes
        )

    env_type = baseline_registry.get_env(config.ENV_NAME)
    env = env_type(config, dataset)
    env.seed(config.RANDOM_SEED)

    return env
