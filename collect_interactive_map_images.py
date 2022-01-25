import os
from typing import Dict, Any
from collections import OrderedDict, defaultdict

import torch
import numpy as np
from tqdm import tqdm

from habitat import get_config
from habitat.core.env import Env
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps

from odometry.config.default import get_config as get_train_config
from odometry.dataset import make_transforms
from odometry.models import make_model

from agent import PPOAgentV2
from habitat_extensions.sensors.egomotion_pg_sensor import PointGoalEstimator
import habitat_extensions.tasks.nav.nav # noqa - required to register TopDownMap

cv2 = try_cv2_import()

POINT_PADDING = 4
xy_points = [
    # row 1:
    [350, 160],
    [547, 140],
    [730,  80],
    # row 2:
    [370, 280],
    [490, 280],
    [745, 260],
    # row 3:
    [360, 555],
    [445, 480],
    [680, 455],
    [847, 430],
    # row 4:
    [380, 670],
    [515, 680],
    [880, 665],
    # row 5:
    [380, 870],
    [515, 880],
    [715, 910],
    [895, 880],
    # row 6:
    [115, 1078],
    [315, 1100],
    [515, 1080],
    [715, 1120],
    [955, 1085],
    # row 7:
    [325, 1300],
    [750, 1240],
    [925, 1330],
]

def colorize_draw_agent(topdown_map_info: Dict[str, Any], episode):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent.

    :param topdown_map_info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    top_down_map = topdown_map_info["map"]
    top_down_map_colorized = maps.colorize_topdown_map(
        top_down_map, topdown_map_info["fog_of_war_mask"]
    )

    source_point_idx, target_point_idx = episode.episode_id.split('x')
    source_point_idx, target_point_idx = int(source_point_idx.lstrip()), int(target_point_idx.lstrip())
    source_point = xy_points[source_point_idx-1]
    target_point = xy_points[target_point_idx-1]

    fog_of_war_desat_amount = 0.85
    fog_of_war_mask = np.zeros_like(top_down_map)

    cv2.circle(fog_of_war_mask, (target_point[0], target_point[1]), 36, 1, -1)

    if fog_of_war_mask is not None:
        fog_of_war_desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        # Only desaturate things that are valid points as only valid points get revealed
        desat_mask = top_down_map != maps.MAP_INVALID_POINT

        top_down_map_colorized[desat_mask] = (
                top_down_map_colorized * fog_of_war_desat_values[fog_of_war_mask]
        ).astype(np.uint8)[desat_mask]

    top_down_map_colorized[
        source_point[1] - POINT_PADDING: source_point[1] + POINT_PADDING + 1,
        source_point[0] - POINT_PADDING: source_point[0] + POINT_PADDING + 1,
    ] = [0, 0, 200]

    top_down_map_colorized[
        target_point[1] - POINT_PADDING: target_point[1] + POINT_PADDING + 1,
        target_point[0] - POINT_PADDING: target_point[0] + POINT_PADDING + 1,
    ] = [200, 0, 0]

    map_agent_pos = topdown_map_info["agent_map_coord"]
    top_down_map = maps.draw_agent(
        image=top_down_map_colorized,
        agent_center_coord=map_agent_pos,
        agent_rotation=topdown_map_info["agent_angle"],
        agent_radius_px=min(top_down_map_colorized.shape[0:2]) // 32,
    )

    if top_down_map_colorized.shape[0] > top_down_map_colorized.shape[1]:
        top_down_map_colorized = np.rot90(top_down_map, 1)

    return top_down_map_colorized


def collect_interactive_map_images():
    dest_dir = '/home/rpartsey/code/3d-navigation/pointgoal-navigation/viz_images/interactive_map_images'
    scene_name = 'Scioto'
    dest_scene_dir = os.path.join(dest_dir, scene_name.lower())
    os.makedirs(dest_scene_dir)

    seed = 1
    gpu_id = 0
    device = torch.device(f'cuda:{gpu_id}')
    task_config_file = 'config_files/challenge_pointnav2021.local.rgbd.yaml'
    ddppo_checkpoint_path = '/home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/ddppo/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31_ckpt.94_spl_0.8003.pth'

    # load VO
    vo_checkpoint_path = '/home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/vo/vo_hc_2021/best_checkpoint_064e.pt'
    vo_config_path = '/home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/vo/vo_hc_2021/config.yaml'
    vo_config = get_train_config(vo_config_path, new_keys_allowed=True)

    vo_model = make_model(vo_config.model).to(device)
    checkpoint = torch.load(vo_checkpoint_path, map_location=device)
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        new_checkpoint[k.replace('module.', '')] = v
    checkpoint = new_checkpoint
    vo_model.load_state_dict(checkpoint)
    vo_model.eval()

    pointgoal_estimator = PointGoalEstimator(
        obs_transforms=make_transforms(vo_config.val.dataset.transforms),
        vo_model=vo_model,
        action_embedding_on=True,
        depth_discretization_on=False,
        rotation_regularization_on=True,
        vertical_flip_on=True,
        device=device
    )

    # load DD-PPO
    checkpoint = torch.load(ddppo_checkpoint_path, map_location=device)
    config = checkpoint["config"]
    config.defrost()
    config.PTH_GPU_ID = gpu_id
    config.RANDOM_SEED = seed
    config.INPUT_TYPE = "depth"
    config.MODEL_PATH = ddppo_checkpoint_path

    config.TASK_CONFIG = get_config(task_config_file)
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz'
    config.TASK_CONFIG.DATASET.SCENES_DIR = 'data/scene_datasets/'
    config.TASK_CONFIG.DATASET.SPLIT = 'interactive_map'
    # config.TASK_CONFIG.DATASET.CONTENT_SCENES = [scene_name]
    config.TASK_CONFIG.TASK.MEASUREMENTS.extend(['TOP_DOWN_MAP', 'COLLISIONS'])
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.LINE_THICKNESS = 8
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.METERS_PER_PIXEL = 0.01
    config.TASK_CONFIG.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = False
    config.freeze()

    agent = PPOAgentV2(config, pointgoal_estimator)

    agg_metrics = defaultdict(float)

    num_val_episodes = 24
    with Env(config=config.TASK_CONFIG) as env:
        episodes_count = 0
        for episode in tqdm(range(num_val_episodes)):
            observations = env.reset()
            agent.reset()

            if not env.current_episode.episode_id.startswith('01x'):
                continue

            current_episode = env.current_episode
            dest_episode_dir = os.path.join(dest_scene_dir, f'episode_{current_episode.episode_id.zfill(2)}')
            os.makedirs(dest_episode_dir)

            dest_rgb_dir = os.path.join(dest_episode_dir, 'rgb')
            dest_depth_dir = os.path.join(dest_episode_dir, 'depth')
            dest_td_map_dir = os.path.join(dest_episode_dir, 'td_map')
            os.makedirs(dest_rgb_dir)
            os.makedirs(dest_depth_dir)
            os.makedirs(dest_td_map_dir)

            step = 0
            while not env.episode_over:
                step += 1

                action = agent.act(observations)

                observations = env.step(action)
                rgb = observations['rgb']
                depth = observations['depth']

                metrics = env.get_metrics()
                td_map_info = metrics.pop('top_down_map')


                # save frames:
                step_str = f'step_{str(step).zfill(3)}'
                cv2.imwrite(os.path.join(dest_rgb_dir, f'{step_str}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

                depth_16bit = (depth.astype(np.float32) * np.iinfo(np.uint16).max).astype(np.uint16)
                cv2.imwrite(os.path.join(dest_depth_dir, f'{step_str}.png'), depth_16bit)

                top_down_map = colorize_draw_agent(td_map_info, current_episode)
                cv2.imwrite(os.path.join(dest_td_map_dir, f'{step_str}.png'), cv2.cvtColor(top_down_map, cv2.COLOR_RGB2BGR))

            metrics.pop('collisions')
            for m, v in metrics.items():
                agg_metrics[m] += v

            episodes_count += 1
            print("Episode finished")

        avg_metrics = {k: v / episodes_count for k, v in agg_metrics.items()}
        print(avg_metrics)


if __name__ == "__main__":
    collect_interactive_map_images()
