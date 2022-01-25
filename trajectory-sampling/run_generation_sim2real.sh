#!/bin/bash
#SBATCH --job-name=odometry_ds
#SBATCH --gres=gpu:8   #gpu:volta:8
#SBATCH --constraint=volta32gb
#SBATCH --nodes 4
#SBATCH --cpus-per-task 80
#SBATCH --ntasks-per-node 1
#SBATCH --mem=450GB #maybe 450, was 500GB
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@600
#SBATCH --partition=learnlab
#SBATCH --open-mode=append
#SBATCH --comment="CVPR 2022 rebbutal 29 Jan"

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

EXP_NAME="generate_odometry_ds"
#--gres gpu:8  # added 32 GB

#$MODULESHOME/init/bash

source ~/.bash_profile
source ~/.profile
source /etc/bash.bashrc
source /etc/profile

module unload cuda
module load cuda/10.1
module unload cudnn
module load cudnn/v7.6.5.32-cuda.10.1
module load anaconda3/5.0.1
module load gcc/7.1.0
module load cmake/3.10.1/gcc.5.4.0
source activate challenge_2021

export CUDA_HOME="/public/apps/cuda/10.1"
export CUDA_NVCC_EXECUTABLE="/public/apps/cuda/10.1/bin/nvcc"
export CUDNN_INCLUDE_PATH="/public/apps/cuda/10.1/include/"
export CUDNN_LIBRARY_PATH="/public/apps/cuda/10.1/lib64/"
export LIBRARY_PATH="/public/apps/cuda/10.1/lib64"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1

CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
echo $CUDA_VISIBLE_DEVICES

CMD_OPTS=$(cat "$CMD_OPTS_FILE")

set -x

srun python -u ./generate_trajectory_dataset_par.py \
  --agent-type spf \
  --data-dir /checkpoint/maksymets/data/vo_dataset_sim2real \
  --config-file ../config_files/shortest_path_follower/shortest_path_follower.yaml \
  --base-task-config-file ../config_files/challenge_pointnav2021.local.rgbd.yaml \
  --dataset gibson \
  --split train \
  --num-episodes-per-scene 4000 \
  --gpu-ids 0 1 2 3 4 5 6 7 \
  --num-processes-per-gpu 4 \
  --pts-frac-per-episode 0.2
