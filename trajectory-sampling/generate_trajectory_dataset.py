import os
import json
import random
import argparse
import subprocess
import quaternion
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from agent import build_agent
from environment import build_env
from config import cfg_model, cfg_rl
from habitat_extensions.config.default import get_config as cfg_env

action_map_old = {
    0: "MOVE_FORWARD",
    1: "TURN_LEFT",
    2: "TURN_RIGHT",
    3: "STOP"
}
action_map_new = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT"
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/datasets/extra_space2/rpartsey/3d-navigation/habitat/pointnav-egomotion/vo/trajectory-noisy",
        help="root directory for writing data/stats",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gibson",
        choices=["gibson", "mp3d"],
        help="dataset (gibson, mp3d)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="dataset split"
    )
    parser.add_argument(
        "--actuation-type",
        type=str,
        default="noiseless",
        choices=["noiseless", "noisy"],
        help="noiseless/noisy actuations",
    )
    parser.add_argument(
        "--gibson-votes-csv",
        type=str,
        default="data/datasets/pointnav/gibson/v1/gibson_quality_ratings.csv",
        help="file with human votes for Gibson envs",
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
        "--sim-gpu-id",
        type=int,
        default=0,
        help="gpu id on which scenes are loaded",
    )
    parser.add_argument(
        "--pth-gpu-id",
        type=int,
        default=1,
        help="gpu id on which PyTorch tensors are loaded",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        default="../config_files/challenge_pointnav2020.local.rgbd.yaml",
        help="path to config yaml containing information about environment",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="../config_files/trajectory-sampling/model_config.yaml",
        help="path to config yaml containing information about pre-trained models",
    )
    parser.add_argument(
        "--rl-config",
        type=str,
        default="../config_files/trajectory-sampling/rl_config.yaml",
        help="path to config yaml containing information about RL params",
    )
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()
    return args

def load_scene_wise_episode_cnt_stats(args):
    json_path = os.path.join(
        args.data_dir,
        "scene_wise_episode_cnt_stats_{}.json".format(args.split)
    )
    if os.path.isfile(json_path):
        with open(json_path) as f: data = json.load(f)
        return data
    return None

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

if __name__=='__main__':
    
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env_config = cfg_env(config_paths=args.env_config)
    model_config = cfg_model(config_paths=args.model_config)
    rl_config = cfg_rl(config_paths=args.rl_config)

    env = build_env(env_config, rl_config, args)
    agent = build_agent(model_config, args)
    
    args.data_dir = os.path.join(args.data_dir, args.dataset)
    scene_wise_episode_cnt_stats = load_scene_wise_episode_cnt_stats(
        args
    )

    if scene_wise_episode_cnt_stats is None:
        # generate + save the scene-wise episode count info
        # (needed for computing sampling stats)
        
        print("Computing scene-wise episode count stats")

        scene_wise_episode_cnt_stats = {}
        n_episodes = len(env.habitat_env.episodes)
        for i in tqdm(range(n_episodes)):
            observation = env.reset()
            agent.reset()

            curr_episode = env.habitat_env.current_episode
            scene_id = env.habitat_env.current_episode.scene_id
            episode_id = env.habitat_env.current_episode.episode_id

            if scene_id not in scene_wise_episode_cnt_stats:
                scene_wise_episode_cnt_stats[scene_id] = {
                    "episode_ids": [],
                    "n_episodes": -1
                }
            
            scene_wise_episode_cnt_stats[scene_id]["episode_ids"].append(
                episode_id
            )
        
        for _, episode_info in scene_wise_episode_cnt_stats.items():
            episode_info["n_episodes"] = len(
                episode_info["episode_ids"]
            )
        
        json_path = os.path.join(
            args.data_dir,
            "scene_wise_episode_cnt_stats_{}.json".format(args.split)
        )
        with open(json_path, "w") as f:
            json.dump(scene_wise_episode_cnt_stats, f)
    
    else:
        print("Loaded scene-wise episode count stats")

    n_episodes = len(env.habitat_env.episodes)
    scene_name_to_dataset_map = defaultdict(list)
    for i in tqdm(range(n_episodes)):

        # 0. reset env + agent before start of episode
        observation = env.reset()
        agent.reset()

        # 1. fetch the scene and episode id for the episode
        curr_episode = env.habitat_env.current_episode
        scene_id, episode_id = (
            curr_episode.scene_id,
            curr_episode.episode_id
        )

        # 2. make sure that all the directories for
        # saving image frames/depth maps are created on disk
        # for this scene
        scene_name = get_scene_name_from_scene_id(scene_id)
        create_folders_for_scene(args, scene_name)


        # 3. get an estimate for the number of data points
        # to sample for this episode
        n_episodes_in_dset_for_this_scene = (
            scene_wise_episode_cnt_stats[scene_id]["n_episodes"]
        )
        n_data_pts_to_sample_for_episode = (
            int(np.ceil(1000./n_episodes_in_dset_for_this_scene))
        )

        # 4. init episode-specific information buffers
        buffer = defaultdict(list)
        buffer["observations"].append(observation)
        buffer["sim_states"].append(
            env.habitat_env.sim.get_agent_state()
        )

        # 5. roll-out episode
        while not env.habitat_env.episode_over:
            # act
            action = agent.act(observation)
            if agent.config.GOAL_SENSOR_UUID == "pointgoal":
                observation, reward, _, info = env.step((action+1) % 4)
            else:
                observation, reward, _, info = env.step(action)

            # update buffers
            buffer["observations"].append(observation)
            buffer["sim_states"].append(
                env.habitat_env.sim.get_agent_state()
            )
            buffer["rewards"].append(reward)
            buffer["actions"].append(action)

        n_steps_for_episode = env.habitat_env._elapsed_steps
        episode_spl = info["spl"]

        # 6. sample data points from within this episode
        idxs = [x for x in range(n_steps_for_episode)]
        random.shuffle(idxs)
        sample_idxs = idxs[:n_data_pts_to_sample_for_episode]

        # make sure that the last action in the sequence doesn't get sampled
        if any([idx == (n_steps_for_episode-1) for idx in sample_idxs]):
            if agent.config.GOAL_SENSOR_UUID == "pointgoal":
                idx_for_stop_action = buffer["actions"].index(3)
            else:
                idx_for_stop_action = buffer["actions"].index(0)

            assert idx_for_stop_action in sample_idxs
            idx_of_stop_in_sampled_idx_list = (
                sample_idxs.index(idx_for_stop_action)
            )

            if n_steps_for_episode > n_data_pts_to_sample_for_episode:
                rejected_idxs = idxs[n_data_pts_to_sample_for_episode:]
                sample_idxs[idx_of_stop_in_sampled_idx_list] = (
                    rejected_idxs[0]
                )
            else:
                del sample_idxs[idx_of_stop_in_sampled_idx_list]


        # 7. for each data point, create a dataset entry
        data_pts_for_curr_episode = []
        for idx in sample_idxs:
            window_size = np.random.randint(1, args.window_size+1)
            if (idx + window_size) >= n_steps_for_episode:
                window_size = (n_steps_for_episode - idx - 1)
            data = {
                "dataset": args.dataset,
                "scene": scene_name,
                "episode_id": episode_id,
                "n_steps": n_steps_for_episode,
                "step_idx": idx,
                "window_size": window_size,
                "spl": episode_spl,
            }

            frame_s, frame_t = (
                buffer["observations"][idx]["rgb"],
                buffer["observations"][idx+window_size]["rgb"],
            )
            depth_s, depth_t = (
                buffer["observations"][idx]["depth"],
                buffer["observations"][idx+window_size]["depth"],
            )
            state_s, state_t = (
                buffer["sim_states"][idx],
                buffer["sim_states"][idx+window_size],
            )
            action = buffer["actions"][idx:(idx+window_size)]
            if agent.config.GOAL_SENSOR_UUID == "pointgoal":
                data["action"] = [action_map_old[act] for act in action]
            else:
                data["action"] = [action_map_new[act] for act in action]

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

    # 8. write all scene jsons to disk
    for scene_name, dataset in scene_name_to_dataset_map.items():
        scene_json_path = os.path.join(
            args.data_dir,
            args.split,
            "{}.json".format(scene_name)
        )
        json_data = {"dataset": dataset}
        with open(scene_json_path, "w") as f: json.dump(json_data, f)

