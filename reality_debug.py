from collections import OrderedDict, defaultdict
from pprint import pprint

import numpy as np
import torch
from habitat import get_config

from habitat.core.env import Env
from habitat.core.utils import try_cv2_import
from habitat_baselines.common.baseline_registry import baseline_registry
from tqdm import tqdm

from agent import PPOAgentV2
from habitat_extensions.sensors.egomotion_pg_sensor import PointGoalEstimator
from odometry.dataset import make_transforms
from odometry.models import make_model
from odometry.config.default import get_config as get_train_config
import habitat_baselines_extensions.common.obs_transformers  # noqa required to register Resize obs transform

cv2 = try_cv2_import()


def reality_debug():
    project_root = '/home/rpartsey/code/3d-navigation/pointgoal-navigation'
    sim2real_dir = f'{project_root}/checkpoints/sim2real'
    pointnav_dir = f'{project_root}/pointgoal-navigation'
    task_config_file = f'{pointnav_dir}/config_files/challenge_pointnav2021_gt_loc_sim2real_hfov.local.rgbd.yaml'
    ddppo_checkpoint_path = f'{sim2real_dir}/sim2real.ckpt.66.pth'
    vo_config_path = f'{project_root}/experiments/sim2real/resnet50_bs32_rgbd_actemb2_flip_invrot/eval_config.yaml'
    vo_checkpoint_path = f'{project_root}/experiments/sim2real/resnet50_bs32_rgbd_actemb2_flip_invrot/best_checkpoint_047e.pt'

    # load VO
    # device = torch.device('cpu')
    pth_gpu_id = 1
    device = torch.device(f'cuda:{pth_gpu_id}')
    vo_config = get_train_config(vo_config_path, new_keys_allowed=True)

    vo_model = make_model(vo_config.model).to(device)
    checkpoint = torch.load(vo_checkpoint_path, map_location=device)
    new_checkpoint = OrderedDict()
    for k, v in checkpoint.items():
        new_checkpoint[k.replace('module.', '')] = v
    checkpoint = new_checkpoint
    vo_model.load_state_dict(checkpoint)
    vo_model.eval()

    action_embedding_on = vo_config.model.params.action_embedding_size > 0
    flip_on = True
    swap_on = True
    depth_discretization_on = False

    pointgoal_estimator = PointGoalEstimator(
        obs_transforms=make_transforms(vo_config.val.dataset.transforms),
        vo_model=vo_model,
        action_embedding_on=action_embedding_on,
        depth_discretization_on=depth_discretization_on,
        rotation_regularization_on=swap_on,
        vertical_flip_on=flip_on,
        device=device
    )

    # load DD-PPO
    checkpoint = torch.load(ddppo_checkpoint_path, map_location=device)
    config = checkpoint["config"]
    config.defrost()
    config.PTH_GPU_ID = pth_gpu_id
    config.RANDOM_SEED = 1
    config.INPUT_TYPE = "depth"
    config.MODEL_PATH = ddppo_checkpoint_path
    config.RL.POLICY.OBS_TRANSFORMS.RESIZE.SIZE = 256

    config.TASK_CONFIG = get_config(task_config_file)
    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.DATASET.DATA_PATH = 'data/datasets/pointnav/gibson/v2/{split}/{split}.json.gz'
    config.TASK_CONFIG.DATASET.SCENES_DIR = 'data/scene_datasets/'
    config.TASK_CONFIG.DATASET.SPLIT = 'val'
    config.freeze()
    agent = PPOAgentV2(config, pointgoal_estimator)

    with Env(config=config.TASK_CONFIG) as env:
        agg_metrics = defaultdict(float)

        num_episodes = 5  # len(env.episodes)
        for _ in tqdm(range(num_episodes)):
            agent.reset()
            observations = env.reset()

            while not env.episode_over:
                # if "rgb" in observations:
                #     observations["rgb"] = observations["rgb"].astype(np.float32) / 255.0
                action = agent.act(observations)
                observations = env.step(action)

            metrics = env.get_metrics()
            for m, v in metrics.items():
                agg_metrics[m] += v

        avg_metrics = {k: v / num_episodes for k, v in agg_metrics.items()}
        pprint(avg_metrics)


if __name__ == "__main__":
    reality_debug()
