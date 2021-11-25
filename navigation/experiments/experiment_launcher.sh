#!/bin/bash
set -x
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";

BASE_TASK_CONFIG_PATH="config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml"
TRAINER_CONFIG="config_files/ddppo/ddppo_pointnav_2021.yaml"

#EXP_NAME="pointnav2021_gt_loc_rgbd_${CURRENT_DATETIME}"
#DATASET_CONTENT_SCENES=""
#MAX_SCENE_REPEAT_STEPS=""
#NUM_EPISODE_SAMPLE=""
#SENSORS=""


#EXP_NAME="pointnav2021_gt_loc_rgbd_${CURRENT_DATETIME}" #
#DATASET_CONTENT_SCENES="TASK_CONFIG.SEED 7 RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights data/new_checkpoints/pointnav2021_gt_loc_rgbd_2021_03_23_14_59_34/ckpt.6.pth"


#EXP_NAME="pointnav2021_gt_loc_rgbd_suc_dist_0.18_${CURRENT_DATETIME}" #
#DATASET_CONTENT_SCENES="TASK_CONFIG.TASK.SUCCESS.SUCCESS_DISTANCE 0.18 TASK_CONFIG.SEED 7 RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights /private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_rgbd_2021_03_23_22_43_10/ckpt.345.pth"

# Gibson 0
#EXP_NAME="pointnav2021_gt_loc_gibson0_pretrained_${CURRENT_DATETIME}" #
#DATASET_CONTENT_SCENES="TASK_CONFIG.DATASET.SPLIT train_extra_large TASK_CONFIG.DATASET.DATA_PATH /checkpoint/sramakri/projects/Habitat-Matterport/pointnav_datasets/gibson/v2/{split}/{split}.json.gz   BASE_TASK_CONFIG_PATH config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml TASK_CONFIG.SEED 7 RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights /private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_rgbd_2021_03_23_22_43_10/ckpt.345.pth"

#EXP_NAME="pointnav2021_gt_loc_gibson0_pretrained_spl_rew${CURRENT_DATETIME}" #
#DATASET_CONTENT_SCENES="TASK_CONFIG.DATASET.SPLIT train_extra_large TASK_CONFIG.DATASET.DATA_PATH /checkpoint/sramakri/projects/Habitat-Matterport/pointnav_datasets/gibson/v2/{split}/{split}.json.gz   BASE_TASK_CONFIG_PATH config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml TASK_CONFIG.SEED 7 RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights /private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_gibson0_pretrained_2021_09_22_13_12_08/ckpt.493.pth"
#TRAINER_CONFIG="config_files/ddppo/ddppo_pointnav_2021_spl_reward.yaml"

#EXP_NAME="pointnav2021_gt_loc_gibson0_pretrained_spl_rew_trsh${CURRENT_DATETIME}" #
#DATASET_CONTENT_SCENES="TASK_CONFIG.DATASET.SPLIT train_extra_large TASK_CONFIG.DATASET.DATA_PATH /checkpoint/sramakri/projects/Habitat-Matterport/pointnav_datasets/gibson/v2/{split}/{split}.json.gz   BASE_TASK_CONFIG_PATH config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml TASK_CONFIG.SEED 7 
#RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights /private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31/ckpt.22.pth"
#TRAINER_CONFIG="config_files/ddppo/ddppo_pointnav_2021_spl_thrsh_reward.yaml"


# SIM2REAL Nav policy
#EXP_NAME="pointnav2021_gt_loc_gibson0_pretr_spl_rew_s2r${CURRENT_DATETIME}" #
#BASE_TASK_CONFIG_PATH="config_files/challenge_pointnav2021_gt_loc_sim2real.local.rgbd.yaml"
#DATASET_CONTENT_SCENES="TASK_CONFIG.DATASET.SPLIT train_extra_large TASK_CONFIG.DATASET.DATA_PATH /checkpoint/sramakri/projects/Habitat-Matterport/pointnav_datasets/gibson/v2/{split}/{split}.json.gz   BASE_TASK_CONFIG_PATH config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml TASK_CONFIG.SEED 7 RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights /private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31/ckpt.62.pth"
#TRAINER_CONFIG="config_files/ddppo/ddppo_pointnav_2021_spl_reward.yaml"


# SIM2REAL Nav policy hfov
EXP_NAME="pointnav2021_gt_loc_gibson0_pretr_spl_rew_s2r_hfov${CURRENT_DATETIME}" #
BASE_TASK_CONFIG_PATH="config_files/challenge_pointnav2021_gt_loc_sim2real_hfov.local.rgbd.yaml"
DATASET_CONTENT_SCENES="TASK_CONFIG.DATASET.SPLIT train_extra_large TASK_CONFIG.DATASET.DATA_PATH /checkpoint/sramakri/projects/Habitat-Matterport/pointnav_datasets/gibson/v2/{split}/{split}.json.gz   BASE_TASK_CONFIG_PATH config_files/challenge_pointnav2021_gt_loc.local.rgbd.yaml TASK_CONFIG.SEED 7 RL.DDPPO.pretrained True RL.DDPPO.pretrained_weights /private/home/maksymets/pointgoal-navigation/data/new_checkpoints/pointnav2021_gt_loc_gibson0_pretrained_spl_rew2021_10_13_02_03_31/ckpt.62.pth"
TRAINER_CONFIG="config_files/ddppo/ddppo_pointnav_2021_spl_reward.yaml"


MAX_SCENE_REPEAT_STEPS=""
NUM_EPISODE_SAMPLE=""
SENSORS=""






LOG_DIR="/checkpoint/maksymets/logs/habitat_baselines/ddppo/pointgoal_nav/${EXP_NAME}"
CHKP_DIR="data/new_checkpoints/${EXP_NAME}"
CMD_OPTS_FILE="${LOG_DIR}/cmd_opt.txt"

CMD_EVAL_OPTS="--exp-config ${TRAINER_CONFIG} BASE_TASK_CONFIG_PATH $BASE_TASK_CONFIG_PATH EVAL_CKPT_PATH_DIR ${CHKP_DIR} CHECKPOINT_FOLDER ${CHKP_DIR} TENSORBOARD_DIR ${LOG_DIR} ${RL_PPO_NUM_STEPS} ${SENSORS}"
CMD_OPTS="${CMD_EVAL_OPTS} ${DATASET_CONTENT_SCENES} ${MAX_SCENE_REPEAT_STEPS} ${NUM_EPISODE_SAMPLE}"


mkdir -p ${CHKP_DIR}
mkdir -p ${LOG_DIR}
echo "$CMD_OPTS" > ${CMD_OPTS_FILE}

sbatch --export=ALL,CMD_OPTS_FILE=${CMD_OPTS_FILE} --job-name=${EXP_NAME: -8} --output=$LOG_DIR/log.out --error=$LOG_DIR/log.err navigation/experiments/run_experiment.sh


CMD_EVAL_OPTS_FILE="${LOG_DIR}/cmd_eval_opt.txt"
CMD_EVAL_OPTS="${CMD_EVAL_OPTS} EVAL.SPLIT val TASK_CONFIG.DATASET.CONTENT_SCENES [\"*\",\"*\"] VIDEO_OPTION []"
echo "$CMD_EVAL_OPTS" > ${CMD_EVAL_OPTS_FILE}
# val on new episodes [\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\"]
sbatch --export=ALL,CMD_OPTS_FILE=${CMD_EVAL_OPTS_FILE} --job-name=${EXP_NAME: -7}e --output=$LOG_DIR/log_eval.out --error=$LOG_DIR/log_eval.err navigation/experiments/run_experiment_eval.sh










#sbatch --export=ALL,CMD_OPTS_FILE=/checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56/cmd_eval_opt.txt --job-name=3_25_56e --output=/checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56/log_eval.out --error=/checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56/log_eval.err experiments/run_obj_nav_eval.sh
#BASE_TASK_CONFIG_PATH configs/tasks/objectnav_mp3d.yaml EVAL_CKPT_PATH_DIR data/new_checkpoints/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56 CHECKPOINT_FOLDER data/new_checkpoints/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56 TENSORBOARD_DIR /checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56   TEST_EPISODE_COUNT 2195 EVAL.SPLIT val_mini VIDEO_OPTION []


# /private/home/maksymets/python_wrapper.sh -u /private/home/maksymets/habitat-lab-pr/habitat_baselines/run.py --exp-config habitat_baselines/config/objectnav/ddppo_objectnav.yaml --run-type eval BASE_TASK_CONFIG_PATH configs/tasks/objectnav_mp3d_256.yaml EVAL_CKPT_PATH_DIR data/new_checkpoints/obj_nav_mp3d_all_train_depth_sem_cat_no_up_down2020_05_27_15_32_03/ckpt.106.pth CHECKPOINT_FOLDER data/new_checkpoints/obj_nav_mp3d_all_train_depth_sem_cat_no_up_down2020_05_27_15_32_03 TENSORBOARD_DIR /checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_depth_sem_cat_no_up_down2020_05_27_15_32_03  SENSORS "[\"DEPTH_SENSOR\",\"SEMANTIC_SENSOR\"]" TASK_CONFIG.TASK.SENSORS [\"COMPASS_SENSOR\",\"GPS_SENSOR\",\"OBJECTSEMANTIC_SENSOR\"] EVAL.SPLIT val_mini NUM_PROCESSES 1 RL.PPO.num_mini_batch 1 TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION 12500
