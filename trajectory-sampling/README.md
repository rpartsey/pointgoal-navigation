# Visual odometry dataset generation

Visual odometry dataset is collected by sampling pairs of RGBD observations (and additional meta information,
see `generate_trajectory_dataset.py`) from agent rollout trajectories.

Add the _pointgoal-navigation_ root directory to the _PYTHONPATH_ before running `./generate_dataset.sh`.
```shell
export PYTHONPATH="<pointgoal-navigation root directory>:${PYTHONPATH}"
```
Add _COLLISIONS_ measure to the base task config _TASK.MEASUREMENTS_ if it is missing.

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
