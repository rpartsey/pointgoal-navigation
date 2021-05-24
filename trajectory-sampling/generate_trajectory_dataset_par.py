import copy
import multiprocessing
import os
import json
import random
import argparse
import subprocess
from pprint import pprint

import quaternion
import numpy as np
from PIL import Image
from tqdm.auto import tqdm, trange
from collections import defaultdict
from itertools import groupby

from agent import DDPPOAgent, ShortestPathFollowerAgent
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from environment.build_env import get_gibson_scenes

# Disable logging to prevent tqdm progressbar corruption
import logging
logging.disable(logging.CRITICAL)

ACTION_MAP = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT"
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-type",
        required=True,
        type=str,
        choices=['ddppo', 'spf'],
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="path to the directory where the generated dataset will be stored",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        type=str,
        help="path to the agent configuration file",
    )
    parser.add_argument(
        "--base-task-config-file",
        required=True,
        type=str,
        help="path to the task configuration file "
             "(if agent-type is ddppo then config should contain "
             "POINTGOAL_WITH_GPS_COMPASS_SENSOR sensor & COLLISIONS measure)",
    )
    parser.add_argument(
        "--model-path",
        required=False,
        type=str,
        help="path to the ddppo checkpoint if agent-type is ddppo"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=["gibson", "mp3d"],
        help="dataset (gibson, mp3d)"
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        choices=["train", "val", "val_mini"],
        help="dataset split"
    )
    parser.add_argument(
        "--gibson-votes-csv",
        required=True,
        type=str,
        help="path to the gibson_quality_ratings.csv",
    )
    parser.add_argument(
        "--num-episodes-per-scene",
        required=True,
        type=int,
        help="number of episodes to sample",
    )
    parser.add_argument(
        "--pts-frac-per-episode",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--max-pts-per-scene",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--single-scene-test",
        action='store_true',
        help="whether to test for episodes of a single, random scene",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="the window size from within which sample source/target for generalized dset",
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        default=[0],
        nargs='+',
        help="gpu ids on which scenes are loaded",
    )
    parser.add_argument(
        "--num-processes-per-gpu",
        type=int,
        default=1,
        help="number of workers to run per GPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100
    )
    args = parser.parse_args()

    return args


def get_scene_name_from_scene_id(scene_id):
    return scene_id.strip().split("/")[-1].split(".")[0]


def get_scene_folder_path_from_scene_name(args, scene_name):
    return os.path.join(
        args.data_dir,
        args.split,
        scene_name
    )


def create_folders_for_scene(args, scene_name):
    # args.data_dir already has dataset name appended to it
    dir = get_scene_folder_path_from_scene_name(args, scene_name)
    if os.path.isdir(dir):
        return
    else:
        # os.makedirs(os.path.join(dir, "rgb"), exist_ok=True)
        # os.makedirs(os.path.join(dir, "depth"), exist_ok=True)

        mkdir_scene_cmd = "mkdir -p {}".format(dir)
        subprocess.check_output(mkdir_scene_cmd, shell=True)

        mkdir_rgb_cmd = "mkdir -p {}".format(os.path.join(
            dir,
            "rgb"
        ))
        mkdir_depth_cmd = "mkdir -p {}".format(os.path.join(
            dir,
            "depth"
        ))
        subprocess.check_output(mkdir_rgb_cmd, shell=True)
        subprocess.check_output(mkdir_depth_cmd, shell=True)


def get_frame_paths(
    args,
    scene_name,
    episode_id,
    idx,
    action):

    return (
        os.path.join(
            args.data_dir,
            args.split,
            scene_name,
            "rgb",
            "{:03d}_{:03d}_{:s}_{:s}.jpg".format(
                int(episode_id), idx, "_".join(action), "source"
            )
        ),
        os.path.join(
            args.data_dir,
            args.split,
            scene_name,
            "rgb",
            "{:03d}_{:03d}_{:s}_{:s}.jpg".format(
                int(episode_id), idx, "_".join(action), "target"
            )
        ),
    )


def get_depth_map_paths(
    args,
    scene_name,
    episode_id,
    idx,
    action):

    return (
        os.path.join(
            args.data_dir,
            args.split,
            scene_name,
            "depth",
            "{:03d}_{:03d}_{:s}_{:s}_depth.npy".format(
                int(episode_id), idx, "_".join(action), "source"
            )
        ),
        os.path.join(
            args.data_dir,
            args.split,
            scene_name,
            "depth",
            "{:03d}_{:03d}_{:s}_{:s}_depth.npy".format(
                int(episode_id), idx, "_".join(action), "target"
            )
        ),
    )


def build_env(config):
    dataset = PointNavDatasetV1(config.TASK_CONFIG.DATASET)
    env_type = baseline_registry.get_env(config.ENV_NAME)
    env = env_type(config, dataset)
    env.seed(config.RANDOM_SEED)

    return env


def collect_scene_dataset(params):
    args, config = params
    scene_name = config.TASK_CONFIG.DATASET.CONTENT_SCENES[0]

    env = build_env(config)
    agent = DDPPOAgent(config) if args.agent_type == 'ddppo' else ShortestPathFollowerAgent(
        env=env,
        goal_radius=config.TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE
    )

    nav_episodes = env.habitat_env.episode_iterator.episodes
    scene_wise_episode_cnt_stats = {
        scene_id: {'n_episodes': len(list(scene_episodes))}
        for scene_id, scene_episodes in groupby(nav_episodes, lambda e: e.scene_id)
    }

    scene_name_to_dataset_map = defaultdict(list)
    for i in trange(
            len(nav_episodes),
            desc=scene_name.ljust(20)
    ):
        # 0. reset env + agent before start of episode
        observation = env.reset()
        agent.reset()

        # 1. fetch the scene and episode id for the episode
        curr_episode = env.habitat_env.current_episode
        scene_id, episode_id = (
            curr_episode.scene_id,
            curr_episode.episode_id
        )

        # 2. init episode-specific information buffers
        buffer = defaultdict(list)
        buffer["observations"].append(observation)
        buffer["sim_states"].append(env.habitat_env.sim.get_agent_state())

        # 3. roll-out episode
        while not env.habitat_env.episode_over:
            action = agent.act(observation)
            observation, reward, _, info = env.step(action=action)

            # update buffers
            buffer["actions"].append(action)
            buffer["observations"].append(observation)
            buffer["sim_states"].append(env.habitat_env.sim.get_agent_state())
            buffer["collisions"].append(info['collisions']['is_collision'])

        # episode stats:
        n_episode_steps = env.habitat_env._elapsed_steps
        episode_stats = {
            'spl': info['spl'],
            'softspl': info['softspl'],
            'distance_to_goal': info['distance_to_goal'],
            'n_steps': n_episode_steps,
            'n_collisions': info['collisions']['count']
        }

        # 4. make sure that all the directories for
        # saving image frames/depth maps are created on disk
        # for this scene
        scene_name = get_scene_name_from_scene_id(scene_id)
        create_folders_for_scene(args, scene_name)

        # 5. get an estimate for the number of data points
        # to sample for this episode
        n_pts_to_sample = int(np.ceil(args.pts_frac_per_episode * n_episode_steps))

        if args.max_pts_per_scene != -1:
            n_episodes_for_this_scene = scene_wise_episode_cnt_stats[scene_id]["n_episodes"]
            n_pts_to_sample = int(np.ceil(args.max_pts_per_scene / n_episodes_for_this_scene))

        # 6. sample data points from within this episode
        idxs = list(range(n_episode_steps - 1))  # subtract 1 to not sample last index
        random.shuffle(idxs)
        sample_idxs = idxs[:n_pts_to_sample]

        # 7. for each data point, create a dataset entry
        data_pts_for_curr_episode = []
        for idx in sample_idxs:
            window_size = np.random.randint(1, args.window_size + 1)
            if (idx + window_size) >= n_episode_steps:
                window_size = (n_episode_steps - idx - 1)
            data = {
                "dataset": args.dataset,
                "scene": scene_name,
                "episode_id": episode_id,
                "n_steps": n_episode_steps,
                "step_idx": idx,
                "window_size": window_size,
                "collision": buffer["collisions"][idx],
                **episode_stats
            }

            frame_s, frame_t = (
                buffer["observations"][idx]["rgb"],
                buffer["observations"][idx + window_size]["rgb"],
            )
            depth_s, depth_t = (
                buffer["observations"][idx]["depth"],
                buffer["observations"][idx + window_size]["depth"],
            )
            state_s, state_t = (
                buffer["sim_states"][idx],
                buffer["sim_states"][idx + window_size],
            )

            actions = buffer["actions"][idx:(idx + window_size)]
            data["action"] = [ACTION_MAP[action] for action in actions]

            (
                data["source_frame_path"],
                data["target_frame_path"]
            ) = (
                get_frame_paths(
                    args,
                    scene_name,
                    episode_id,
                    idx,
                    data["action"]
                )
            )
            (
                data["source_depth_map_path"],
                data["target_depth_map_path"]
            ) = (
                get_depth_map_paths(
                    args,
                    scene_name,
                    episode_id,
                    idx,
                    data["action"]
                )
            )

            data["source_agent_state"] = {
                "position": state_s.position.tolist(),
                "rotation": quaternion.as_float_array(
                    state_s.rotation
                ).tolist(),
            }
            data["target_agent_state"] = {
                "position": state_t.position.tolist(),
                "rotation": quaternion.as_float_array(
                    state_t.rotation
                ).tolist(),
            }

            Image.fromarray(frame_s).save(data["source_frame_path"])
            Image.fromarray(frame_t).save(data["target_frame_path"])
            np.save(data["source_depth_map_path"], depth_s)
            np.save(data["target_depth_map_path"], depth_t)

            data_pts_for_curr_episode.append(data)

        scene_name_to_dataset_map[scene_name] += data_pts_for_curr_episode
            # pbar.update()
    env.close()

    # 8. write all scene jsons to disk
    for scene_name, dataset in scene_name_to_dataset_map.items():
        scene_json_path = os.path.join(
            args.data_dir,
            args.split,
            "{}.json".format(scene_name)
        )
        json_data = {"dataset": dataset}
        with open(scene_json_path, "w") as f:
            json.dump(json_data, f)

    return scene_name, len(scene_name_to_dataset_map[scene_name])


if __name__ == '__main__':
    
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    random.seed(args.seed)
    np.random.seed(args.seed)

    config = get_config(
        args.config_file, ["BASE_TASK_CONFIG_PATH", args.base_task_config_file]
    )
    # override config values with command line arguments:
    config.defrost()
    config.INPUT_TYPE = 'rgbd'
    config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = args.seed
    config.TASK_CONFIG.DATASET.SPLIT = args.split
    config.TASK_CONFIG.DATASET.SINGLE_SCENE_TEST = args.single_scene_test
    config.TASK_CONFIG.DATASET.GIBSON_VOTES_CSV = args.gibson_votes_csv
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = args.num_episodes_per_scene
    config.freeze()

    mp_ctx = multiprocessing.get_context("fork")
    NUM_GPUS = len(args.gpu_ids)
    NUM_PROCESSES_PER_GPU = args.num_processes_per_gpu
    NUM_WORKERS = NUM_GPUS * NUM_PROCESSES_PER_GPU

    scene_names = [scene_name for scene_name, _ in get_gibson_scenes(config.TASK_CONFIG)]
    params = [(copy.deepcopy(args), copy.deepcopy(config)) for _ in range(len(scene_names))]

    for i, ((_, worker_config), scene_name) in enumerate(zip(params, scene_names)):
        gpu_id = args.gpu_ids[i % NUM_GPUS] if NUM_GPUS > 1 else args.gpu_ids[0]

        worker_config.defrost()
        worker_config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_name]
        worker_config.TORCH_GPU_ID = gpu_id
        worker_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id
        worker_config.freeze()

    with mp_ctx.Pool(
            NUM_WORKERS,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),),
            maxtasksperchild=1
    ) as pool:
        stats = {}
        with tqdm(total=len(params), desc='Overall progress'.ljust(20)) as pbar:
            for scene_name, num_obs_pairs in pool.imap_unordered(collect_scene_dataset, params):
                stats[scene_name] = num_obs_pairs
                pbar.update()
        pprint(stats)
