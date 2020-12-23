import sys
import csv
import ipdb
import random

from . import NavRLEnv
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


def get_gibson_scenes(env_config, args):
    # get all scenes
    all_gibson_scenes = PointNavDatasetV1.get_scenes_to_load(
        env_config.DATASET
    )

    # filter by votes (only high-quality envs)
    filtered_gibson_scenes = filter_gibson_scenes_by_votes(
        env_config,
        all_gibson_scenes,
        args.gibson_votes_csv
    )
    assert len(filtered_gibson_scenes) > 0
    return filtered_gibson_scenes

def filter_gibson_scenes_by_votes(
    env_config,
    all_scenes,
    votes_csv_file,
    votes_thresh=4.0):
    csv_file = csv.reader(open(votes_csv_file, "r"))

    _ = next(csv_file)
    gibson_scene_to_votes_map = {
        row[0]: float(row[1])
        for row in csv_file
    }

    if env_config.DATASET.SPLIT in ['val']:
        all_scenes = [
            scene.replace(".glb", "")
            for scene in all_scenes
        ]
        filtered_gibson_scenes = [
            ("{}.glb".format(scene), gibson_scene_to_votes_map[scene])
            for scene in all_scenes
            if gibson_scene_to_votes_map[scene] >= votes_thresh
        ]
    else:
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
    env_config,
    filtered_gibson_scenes):
    
    # randomly sample a scene
    random.shuffle(filtered_gibson_scenes)
    scene, _ = filtered_gibson_scenes[0]
    
    env_config = update_config_with_scene_names(
        env_config,
        scene
    )
    dataset = PointNavDatasetV1(env_config.DATASET)
    return env_config, dataset

def get_pointnav_episodes_for_all_scenes(
    env_config,
    filtered_gibson_scenes):

    random.shuffle(filtered_gibson_scenes)
    scene_names = [item[0] for item in filtered_gibson_scenes]
    env_config = update_config_with_scene_names(
        env_config,
        scene_names
    )
    dataset = PointNavDatasetV1(env_config.DATASET)
    return env_config, dataset

def build_env(env_config, rl_config, args):
    random.seed(args.seed)
    filtered_gibson_scenes = get_gibson_scenes(env_config, args)

    if args.single_scene_test:
        env_config, dataset = get_pointnav_episodes_for_random_scene(
            env_config,
            filtered_gibson_scenes
        )
    else:
        env_config, dataset = get_pointnav_episodes_for_all_scenes(
            env_config,
            filtered_gibson_scenes
        )
    
    env_config.defrost()
    env_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = args.sim_gpu_id

    if "EGO_PREDICTION_POINTGOAL_SENSOR" in env_config.TASK.SENSORS:
        sys.path.append("../../odometry")
        from rl.sensors import PointGoalSensorWithEgoPredictions

        env_config.TASK.EGO_PREDICTION_POINTGOAL_SENSOR.MODEL.GPU_ID = (
            args.pth_gpu_id
        )
    env_config.freeze()
    
    env = NavRLEnv(
        config_env=env_config,
        config_baseline=rl_config,
        dataset=dataset
    )
    env.seed(args.seed)

    return env