import copy
import multiprocessing
import os
import json
import gzip
import random
import argparse
from pprint import pprint

import cv2
import quaternion
import numpy as np
from tqdm.auto import tqdm, trange
from collections import defaultdict
from itertools import groupby

from agent import DDPPOAgent, ShortestPathFollowerAgent
from habitat_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1


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
        help="dataset (gibson, mp3d)"
    )
    parser.add_argument(
        "--split",
        required=True,
        type=str,
        help="dataset split (train, val, val_mini)"
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


def create_folders_for_scene(args, scene_name):
    # args.data_dir already has dataset name appended to it
    scene_rgb_dir = os.path.join(args.data_dir, args.split, 'rgb', scene_name)
    os.makedirs(scene_rgb_dir, exist_ok=False)

    scene_depth_dir = os.path.join(args.data_dir, args.split, 'depth', scene_name)
    os.makedirs(scene_depth_dir, exist_ok=False)


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
            "rgb",
            scene_name,
            "{:03d}_{:03d}_{:s}_{:s}.png".format(int(episode_id), idx, "_".join(action), "source")
        ),
        os.path.join(
            args.data_dir,
            args.split,
            "rgb",
            scene_name,
            "{:03d}_{:03d}_{:s}_{:s}.png".format(int(episode_id), idx, "_".join(action), "target")
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
            "depth",
            scene_name,
            "{:03d}_{:03d}_{:s}_{:s}_depth.png".format(int(episode_id), idx, "_".join(action), "source")
        ),
        os.path.join(
            args.data_dir,
            args.split,
            "depth",
            scene_name,
            "{:03d}_{:03d}_{:s}_{:s}_depth.png".format(int(episode_id), idx, "_".join(action), "target")
        ),
    )


def build_env(config):
    dataset = PointNavDatasetV1(config.TASK_CONFIG.DATASET)
    env_type = baseline_registry.get_env(config.ENV_NAME)
    env = env_type(config, dataset)
    env.seed(config.TASK_CONFIG.SEED)

    return env


def collect_dataset(params):
    args, config = params

    stats = {}
    for scene in config.TASK_CONFIG.DATASET.CONTENT_SCENES:
        create_folders_for_scene(args, scene)

        config_copy = config.clone()
        config_copy.defrost()
        config_copy.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene]
        config_copy.freeze()

        scene_name, num_obs_pairs = collect_scene_dataset(args, config_copy)
        stats[scene_name] = num_obs_pairs

    return stats


def collect_scene_dataset(args, config):
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

        # 4. get an estimate for the number of data points
        # to sample for this episode
        n_pts_to_sample = int(np.ceil(args.pts_frac_per_episode * n_episode_steps))

        if args.max_pts_per_scene != -1:
            n_episodes_for_this_scene = scene_wise_episode_cnt_stats[scene_id]["n_episodes"]
            n_pts_to_sample = int(np.ceil(args.max_pts_per_scene / n_episodes_for_this_scene))

        # 5. sample data points from within this episode
        idxs = list(range(n_episode_steps - 1))  # subtract 1 to not sample last index
        random.shuffle(idxs)
        sample_idxs = idxs[:n_pts_to_sample]

        # 6. for each data point, create a dataset entry
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

            dest_size = (320, 180)

            frame_s_resized = cv2.resize(frame_s, dest_size, interpolation=cv2.INTER_LINEAR)
            frame_t_resized = cv2.resize(frame_t, dest_size, interpolation=cv2.INTER_LINEAR)

            frame_s_resized_bgr = cv2.cvtColor(frame_s_resized, cv2.COLOR_RGB2BGR)
            frame_t_resized_bgr = cv2.cvtColor(frame_t_resized, cv2.COLOR_RGB2BGR)

            cv2.imwrite(data['source_frame_path'], frame_s_resized_bgr)
            cv2.imwrite(data['target_frame_path'], frame_t_resized_bgr)

            depth_s_resized = cv2.resize(depth_s, dest_size, interpolation=cv2.INTER_LINEAR)
            depth_t_resized = cv2.resize(depth_t, dest_size, interpolation=cv2.INTER_LINEAR)

            depth_s_resized_16bit = (depth_s_resized.astype(np.float32) * np.iinfo(np.uint16).max).astype(np.uint16)
            depth_t_resized_16bit = (depth_t_resized.astype(np.float32) * np.iinfo(np.uint16).max).astype(np.uint16)

            cv2.imwrite(data['source_depth_map_path'], depth_s_resized_16bit)
            cv2.imwrite(data['target_depth_map_path'], depth_t_resized_16bit)

            data_pts_for_curr_episode.append(data)

        scene_name_to_dataset_map[scene_name] += data_pts_for_curr_episode
            # pbar.update()
    env.close()

    # 7. write all scene jsons to disk
    for scene_name, dataset in scene_name_to_dataset_map.items():
        scene_json_path = os.path.join(
            args.data_dir,
            args.split,
            'json',
            "{}.json.gz".format(scene_name)
        )
        json_data = {"dataset": dataset}
        with gzip.open(scene_json_path, 'wt') as f:
            json.dump(json_data, f)

    return scene_name, len(scene_name_to_dataset_map[scene_name])


if __name__ == '__main__':
    args = parse_args()
    args.data_dir = os.path.join(args.data_dir, args.dataset)

    os.makedirs(os.path.join(args.data_dir, args.split,  'rgb'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, args.split,  'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, args.split, 'json'), exist_ok=True)

    config = get_config(args.config_file, ["BASE_TASK_CONFIG_PATH", args.base_task_config_file])

    if (os.environ.get('SLURM_JOBID') is not None) or (os.environ.get('SLURM_JOB_ID') is not None):
        world_rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        world_rank = 0
        world_size = 1

    num_processes_per_gpu = args.num_processes_per_gpu
    gpu_ids = args.gpu_ids * num_processes_per_gpu
    num_workers = len(gpu_ids)

    args.seed = args.seed + world_rank * num_workers

    random.seed(args.seed)
    np.random.seed(args.seed)

    # override config values with command line arguments:
    config.defrost()
    config.INPUT_TYPE = 'rgbd'
    config.MODEL_PATH = args.model_path
    config.RANDOM_SEED = args.seed
    config.TASK_CONFIG.SEED = args.seed
    config.TASK_CONFIG.DATASET.SPLIT = args.split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = args.num_episodes_per_scene
    if 'COLLISIONS' not in config.TASK_CONFIG.TASK.MEASUREMENTS:
        config.TASK_CONFIG.TASK.MEASUREMENTS.append('COLLISIONS')
    config.freeze()

    json_files = os.listdir(os.path.join(args.data_dir, args.split, 'json'))
    finished_scenes = [file_name.split('.')[0] for file_name in json_files]

    scenes = PointNavDatasetV1.get_scenes_to_load(config.TASK_CONFIG.DATASET)
    scenes = [scene for scene in scenes if scene not in finished_scenes]
    scenes.sort()

    per_distrib_worker = int(np.ceil(len(scenes) / world_size))
    iter_start = world_rank * per_distrib_worker
    iter_stop = min(iter_start + per_distrib_worker, len(scenes))

    distrib_worker_scenes = scenes[iter_start:iter_stop]
    random.shuffle(distrib_worker_scenes)

    scene_splits = [[] for _ in range(min(num_workers, len(distrib_worker_scenes)))]
    for idx, scene in enumerate(distrib_worker_scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    params = []
    for i in range(len(scene_splits)):
        proc_config = config.clone()
        proc_config.defrost()

        proc_config.defrost()
        proc_config.TASK_CONFIG.DATASET.CONTENT_SCENES = scene_splits[i]
        proc_config.TORCH_GPU_ID = gpu_ids[i]
        proc_config.TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_ids[i]
        proc_config.RANDOM_SEED = proc_config.RANDOM_SEED + i
        proc_config.TASK_CONFIG.SEED = proc_config.TASK_CONFIG.SEED + i
        proc_config.freeze()

        params.append((copy.deepcopy(args), proc_config))

    mp_ctx = multiprocessing.get_context("fork")
    with mp_ctx.Pool(
            num_workers,
            initializer=tqdm.set_lock,
            initargs=(tqdm.get_lock(),),
            maxtasksperchild=1
    ) as pool:
        global_stats = {}
        with tqdm(total=len(scenes), desc='Overall progress'.ljust(20)) as pbar:
            for process_stats in pool.imap_unordered(collect_dataset, params):
                for scene_name, num_obs_pairs in process_stats.items():
                    global_stats[scene_name] = num_obs_pairs
                    pbar.update()
            pprint(global_stats)
