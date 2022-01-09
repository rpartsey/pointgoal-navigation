# Visual odometry dataset generation

Visual odometry dataset is collected by sampling pairs of RGB-D observations (and additional meta information,
see `generate_trajectory_dataset_par.py`) from agent rollout trajectories.

Before running `./generate_dataset_par.sh` (or `python generate_trajectory_dataset_par.py`) 
add the _pointgoal-navigation_ root directory to the _PYTHONPATH_:
```shell
export PYTHONPATH="<pointgoal-navigation root directory>:${PYTHONPATH}"
```
and link the habitat _data_ directory:
```shell
ln -s <path to habitat data directory> <path to pointgoal-navigation>/trajectory-sampling/data/
```

## Multi-process

To prevent tqdm progress bar from being corrupted by Habitat's logging 
export following environment variables to disable logging:
```shell
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export KMP_WARNINGS=off
```


### Shortest path follower agent

Example:
```shell
./generate_dataset_par.sh \
  --agent-type spf \
  --data-dir /home/rpartsey/data/habitat/vo_datasets/noisy \
  --config-file ../config_files/shortest_path_follower/shortest_path_follower.yaml \
  --base-task-config-file ../config_files/challenge_pointnav2021.local.rgbd.yaml \
  --dataset gibson \
  --split train \
  --num-episodes-per-scene 4000 \
  --gpu-ids 0 1 \
  --num-processes-per-gpu 4 \
  --pts-frac-per-episode 0.2
```

### DD-PPO agent
When you launch the ddppo agent you should specify also `--model-path` - path to the ddppo checkpoint.

Example:
```shell
./generate_dataset_par.sh \
  --agent-type ddppo \
  --data-dir /home/rpartsey/data/habitat/vo_datasets/noisy \
  --config-file ../config_files/ddppo/ddppo_pointnav_2021.yaml \
  --base-task-config-file ../config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml \
  --model-path /home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/ddppo/pointnav2021_gt_loc_depth_ckpt.345.pth \
  --dataset gibson \
  --split train \
  --gibson-votes-csv data/datasets/pointnav/gibson/v2/gibson_quality_ratings.csv \
  --num-episodes-per-scene 4000 \
  --gpu-ids 0 \
  --num-processes-per-gpu 4 \
  --pts-frac-per-episode 0.2
```

## Single process (deprecated)

### Shortest path follower agent

Example:
```shell
./generate_dataset.sh \
  --agent-type spf \
  --data-dir /home/rpartsey/data/habitat/vo_datasets/noisy \
  --config-file ../config_files/shortest_path_follower/shortest_path_follower.yaml \
  --base-task-config-file ../config_files/challenge_pointnav2021.local.rgbd.yaml \
  --dataset gibson \
  --split train \
  --gibson-votes-csv data/datasets/pointnav/gibson/v2/gibson_quality_ratings.csv \
  --num-episode-sample 15000 \
  --pts-frac-per-episode 0.15
```

### DD-PPO agent
When you launch the ddppo agent you should specify also `--model-path` - path to the ddppo checkpoint.

Example:
```shell
./generate_dataset.sh \
  --agent-type ddppo \
  --data-dir /home/rpartsey/data/habitat/vo_datasets/noisy \
  --config-file ../config_files/ddppo/ddppo_pointnav_2021.yaml \
  --base-task-config-file ../config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml \
  --model-path /home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/ddppo/pointnav2021_gt_loc_depth_ckpt.345.pth \
  --dataset gibson \
  --split train \
  --gibson-votes-csv data/datasets/pointnav/gibson/v2/gibson_quality_ratings.csv \
  --num-episode-sample 15000 \
  --pts-frac-per-episode 0.15
```
