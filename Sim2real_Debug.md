# Sim2real Debug


Performance comparison of different navigation policies and localizations of Gibson val_mini:

| # | spl  |softspl | success | distance_to_goal | agent | localization|train height|eval height| sensors orientation|
| - |   -- |  --  |   -- | --   |  --  |  -- |  --  |  -- | --  |
| 1 | 0.85 | 0.80 | 0.98 | 0.42 | SPF | - |  -  |  - | -20 | 
| 2 | 0.75 | 0.74 | 0.98 | 0.34 | pointnav2021_gt_loc_depth_ckpt.345.pth | GT | 0.88  | 0.88 | -20  |
| 3 | 0.10 | 0.11 | 0.24 | 6.30 | pointnav2021_gt_loc_depth_ckpt.345.pth | GT | 0.88  | 0.88 |   0  |
| 4 | 0.67 | 0.66 | 0.88 | 1.70 | pointnav2021_gt_loc_depth_ckpt.345.pth | GT | 0.88  | 0.61 | -20  |
| 5 | 0.17 | 0.18 | 0.37 | 5.25 | pointnav2021_gt_loc_depth_ckpt.345.pth | GT | 0.88  | 0.61 |   0  |
| 6 | 0.72 | 0.73 | 0.92 | 0.29 | pointnav2021_gt_loc_depth_ckpt.345.pth | VO | 0.88  | 0.88 | -20  |
| 7 | 0.00 | 0.01 | 0.01 | 7.81 | ckpt.89.pth | GT | 1.25  | 0.88 |  -20  |
| 8 | 0.01 | 0.01 | 0.01 | 7.86 | ckpt.89.pth | VO | 1.25  | 0.88 |  -20  |

### Commands
#### SPF 
```
export CHALLENGE_CONFIG_FILE=config_files/sim2real_debug/challenge_pointnav2021.local.rgbd.yaml

python agent.py \
--agent-type ShortestPathFollowerAgent \
--evaluation local
```

#### DD-PPO (GT loc)
```
export CHALLENGE_CONFIG_FILE=config_files/sim2real_debug/challenge_pointnav2021_gt_loc.local.rgbd.yaml

python agent.py \
--agent-type PPOAgentV2 \
--input-type depth \
--evaluation local \
--ddppo-checkpoint-path /home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/ddppo/pointnav2021_gt_loc_depth_ckpt.345.pth \
--ddppo-config-path config_files/sim2real_debug/ddppo_pointnav_2021.yaml \
--pth-gpu-id 0
```

#### DD-PPO + VO
```
export CHALLENGE_CONFIG_FILE=config_files/sim2real_debug/challenge_pointnav2021.local.rgbd.yaml

python agent.py \
--agent-type PPOAgentV2 \
--input-type depth \
--evaluation local \
--ddppo-checkpoint-path /home/rpartsey/code/3d-navigation/pointgoal-navigation/checkpoints/ddppo/pointnav2021_gt_loc_depth_ckpt.345.pth \
--ddppo-config-path config_files/sim2real_debug/ddppo_pointnav_2021.yaml \
--vo-config-path /home/rpartsey/code/3d-navigation/pointgoal-navigation/experiments/mthesis/hc/3mmm/resnet50_bs16_ddepth5_maxd0.5_randomsampling_dropout0.2_poselossv21._1._180x320_embedd_act_vflip_hc2021_vo3_bigdata_3m/config.yaml  \
--vo-checkpoint-path /home/rpartsey/code/3d-navigation/pointgoal-navigation/experiments/mthesis/hc/3mmm/resnet50_bs16_ddepth5_maxd0.5_randomsampling_dropout0.2_poselossv21._1._180x320_embedd_act_vflip_hc2021_vo3_bigdata_3m/best_checkpoint_064e.pt \
--pth-gpu-id 0 \
--rotation-regularization-on \
--vertical-flip-on 
```
