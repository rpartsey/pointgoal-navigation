# Is Mapping Necessary for Realistic PointGoal Navigation? (IMN-RPG)

This repository is the official implementation of 
[Is Mapping Necessary for Realistic PointGoal Navigation?](https://arxiv.org/). 

## Requirements

### Environment
This project is developed with Python 3.6. The recommended way to set up the environment is by using 
[miniconda](https://docs.conda.io/en/latest/miniconda.html) or [anaconda](https://docs.conda.io/en/latest/) 
package/environment management system:

```bash
conda create -n pointgoalnav-env python=3.6 cmake=3.14.0 -y
conda activate pointgoalnav-env
```

IMN-RPG uses [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7) 0.1.7 (commit 856d4b0) which can be [built from source](https://github.com/facebookresearch/habitat-sim/tree/v0.1.7#installation) or installed from conda:

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless
```

Then install [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/challenge-2021):

```bash
git clone --branch challenge-2021 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
# installs both habitat and habitat_baselines
pip install -r requirements.txt
pip install -r habitat_baselines/rl/requirements.txt
pip install -r habitat_baselines/rl/ddppo/requirements.txt
pip install -r habitat_baselines/il/requirements.txt
python setup.py develop --all
```

Now you can install IMN-RPG:

```bash
git clone git@github.com:rpartsey/pointgoal-navigation.git
cd pointgoal-navigation
python -m pip install -r requirements.txt
```

### Data

#### Scenes
Download Gibson and Matterport3D scenes by following the instructions in Habitat-Lab's 
[Data](https://github.com/facebookresearch/habitat-lab/tree/challenge-2021#data) section.

#### Episodes
Download Gibson PointGoal navigation episodes corresponding to Sim2LoCoBot experiment configuration 
(row 2 in [Task dataset](https://github.com/facebookresearch/habitat-lab/tree/challenge-2021#task-datasets) table; 
aka. Gibson-v2 episodes). Matterport3D episodes following the same configuration didn't exist and were generated in scope of IMN-RPG research. 
Train/val episodes can be downloaded here (TODO: add links).


After data downloading create a symlink to `data/` directory in the IMN-RPG project root:
```bash
cd pointgoal-navigation
ln -s <path-to-data-directory> data
```

#### Visual odometry dataset
Visual odometry dataset is collected by sampling pairs of RGB-D observations (and additional information, navigate to 
`trajectory-sampling` see `generate_trajectory_dataset_par.py`) from agent rollout trajectories.

Before running `generate_trajectory_dataset_par.py` add the project root directory to the _PYTHONPATH_:
```shell
export PYTHONPATH="<path-to-pointgoal-navigation-directory>:${PYTHONPATH}"
```
and create a symlink to `data/` directory:
```shell
ln -s <path-to-data-directory> <path-to-pointgoal-navigation-directory>/trajectory-sampling/data/
```

To generate training dataset, run:
```bash
python generate_trajectory_dataset_par.py \
--agent-type spf \
--data-dir /home/rpartsey/data/habitat/vo_datasets/hc_2021 \
--config-file ../config_files/shortest_path_follower/shortest_path_follower.yaml \
--base-task-config-file ../config_files/challenge_pointnav2021.local.rgbd.yaml \
--dataset gibson \
--split train \
--num-episodes-per-scene 2000 \
--pts-frac-per-episode 0.2 \
--gpu-ids 0 1 \
--num-processes-per-gpu 10
```
The above command was used to generate a training dataset (disk space: 592.3 GB, dataset length: 1627439). 
Reported in the paper 0.5M and 1.5M datasets were uniformly sampled from generated dataset.

To generate validation dataset, run:
```bash
python generate_trajectory_dataset_par.py \
--agent-type spf \
--data-dir /home/rpartsey/data/habitat/vo_datasets/hc_2021 \
--config-file ../config_files/shortest_path_follower/shortest_path_follower.yaml \
--base-task-config-file ../config_files/challenge_pointnav2021.local.rgbd.yaml \
--dataset gibson \
--split val \
--num-episodes-per-scene 71 \
--pts-frac-per-episode 0.75 \
--gpu-ids 0 1 \
--num-processes-per-gpu 10
```
The above command was used to generate a validation dataset (disk space: 16.2 GB, dataset length: 44379).


## Training

### Navigation policy
We use policy training pipeline from [habitat_baselines](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines).

See `navigation/experiments/experiment_launcher.sh`

### Visual odometry
Experiment configuration parameters are set in the _yaml_ file. See `config_files/odometry/paper/*`.

To train the visual odometry model, run:
```bash
python train_odometry_v2.py --config-file <path-to-config-file>
```

For multiple GPUs/nodes you may use torch.distributed.launch:
```bash
python -u -m torch.distributed.launch --use_env --nproc_per_node=2 train_odometry_v2.py --config-file <path-to-config-file>
```
or Slurm (see `odometry/experiments/run_experiment.*` files).

## Evaluation
To benchmark the agent (navigation policy + visual odometry), run:

```bash
export CHALLENGE_CONFIG_FILE=config_files/challenge_pointnav2021.local.rgbd.yaml
python agent.py \
--agent-type PPOAgentV2 \
--input-type depth \
--evaluation local \
--ddppo-checkpoint-path <path to policy checkpoint> \
--ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml \
--vo-config-path <path to vo config> \
--vo-checkpoint-path <path to vo checkpoint> \
--pth-gpu-id 0 \
--rotation-regularization-on \
--vertical-flip-on
```

## Pre-trained Models

### Gibson val

[//]: # (rowspan="2")

<table> 
<caption>Table caption</caption> 
<tr> 
    <th rowspan="2"></th>
    <th rowspan="2">Dataset Size(M)</th> 
    <th colspan="2">VO</th> 
    <th colspan="2">Embedding</th> 
    <th colspan="2">Train time</th>
    <th rowspan="2">Epoch</th>
    <th rowspan="2">Model checkpoint</th>
</tr> 
<tr>
    <th>Encoder</th> 
    <th>Size(M)</th> 
    <th>1FC</th> 
    <th>2FC</th>
    <th>Flip</th> 
    <th>Swap</th>
</tr> 
<tr> 
    <td>1</td> 
    <td>0.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td></td> 
    <td></td> 
    <td></td> 
    <td></td>
    <td>50</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>2</td> 
    <td>0.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td></td> 
    <td></td> 
    <td></td>
    <td>43</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>3</td> 
    <td>0.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td></td> 
    <td></td>
    <td>44</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>4</td> 
    <td>0.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td></td>
    <td>&#10004;</td>
    <td>48</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>5</td> 
    <td>0.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td></td>
    <td>50</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>6</td> 
    <td>0.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td>
    <td>50</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>7</td> 
    <td>1.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td></td> 
    <td></td>
    <td>48</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>8</td> 
    <td>1.5</td> 
    <td>ResNet18</td> 
    <td>4.2</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>50</td>
    <td>Link</td>
</tr> 
<tr> 
    <td>9</td> 
    <td>1.5</td> 
    <td>ResNet50</td> 
    <td>7.6</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>48</td>
    <td>Link</td>
</tr> 
</table>


### Reality

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 




[//]: # (# pointgoal-navigation)

[//]: # ()
[//]: # (### Development environment setup)

[//]: # (Development environment and dataset for Gibson PointNav may be set up by running `./pointgoal_navigation_install.sh`.)

[//]: # ()
[//]: # (After the installation process is complete, link the gibson scene dataset to)

[//]: # (the `<path to pointgoal-navigation>/data/scene_datasets`.)

[//]: # ()
[//]: # (```shell)

[//]: # (ln -s <path to scene datasets> <path to pointgoal-navigation>/data/scene_datasets)

[//]: # (```)
