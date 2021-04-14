#!/usr/bin/env bash

set -e

# git clone git@github.com:rpartsey/pointgoal-navigation.git
git checkout main

conda create -n pointgoal-navigation-env python=3.6 cmake=3.14.0 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pointgoal-navigation-env

#   navigate one directory up
cd ..

# ----------------------------------------------------------------------------
# install habitat-sim
# ----------------------------------------------------------------------------

conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch
conda clean -ya
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
git checkout 856d4b08c1a2632626bf0d205bf46471a99502b7 # v0.1.7
python -m pip install -r requirements.txt

module purge
module load cuda/10.0
module load cudnn/v7.4-cuda.10.0
module load cmake/3.15.3/gcc.7.3.0
module load NCCL/2.4.8-1-cuda.10.0
module load gcc/7.3.0
python setup.py install --headless --with-cuda

# silence habitat-sim logs
export GLOG_minloglevel=2
export MAGNUM_LOG="quiet"

cd ..

# ----------------------------------------------------------------------------
# install habitat-lab
# ----------------------------------------------------------------------------

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout ac4339ed7e9bafb5e45fee9c5cf68095f40edee2 # challenge-2021

# install both habitat and habitat_baselines
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt
pip install -r habitat_baselines/rl/ddppo/requirements.txt
pip install -r habitat_baselines/il/requirements.txt
python setup.py develop --all

cd ..

# ----------------------------------------------------------------------------
#   install pointgoal-navigation requirements
# ----------------------------------------------------------------------------
cd pointgoal-navigation

python -m pip install torch==1.7.1 torchvision==0.8.2 segmentation-models-pytorch==0.1.3

# ----------------------------------------------------------------------------
#   download the dataset for Gibson PointNav
# ----------------------------------------------------------------------------
python -m pip install gdown

gdown https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v2/pointnav_gibson_v2.zip
mkdir -p data/datasets/pointnav/gibson
unzip pointnav_gibson_v2.zip -d data/datasets/pointnav/gibson
rm pointnav_gibson_v2.zip
gdown https://drive.google.com/uc?id=15_vh9rZgNhk_B8RFWZqmcW5JRdNQKM2G --output data/datasets/pointnav/gibson/gibson_quality_ratings.csv

NOCOLOR='\033[0m'
RED='\033[0;31m'
echo -e "\n${RED}NOTE:${NOCOLOR} use command 'ln -s <path to scene datasets> ${PWD}/data/scene_datasets' to link the simulation scenes.\n"
