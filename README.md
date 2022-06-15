# Is Mapping Necessary for Realistic PointGoal Navigation? (IMN-RPG)

This repository is the official implementation of 
[Is Mapping Necessary for Realistic PointGoal Navigation?](http://arxiv.org/abs/2206.00997). 

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
Train/val episodes can be downloaded [here](https://drive.google.com/file/d/1XJbx_oV2eK7ZXBTssfv-bij1zTS7Mhz9/view?usp=sharing).


After data downloading create a symlink to `data/` directory in the IMN-RPG project root:
```bash
cd pointgoal-navigation
ln -s <path-to-data-directory> data
```

#### Visual Odometry Dataset
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
--data-dir data/vo_datasets/hc_2021 \
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
--data-dir data/vo_datasets/hc_2021 \
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

### Navigation Policy


We use policy training pipeline from [habitat_baselines](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines).

See `navigation/experiments/experiment_launcher.sh`

### Visual Odometry
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
--ddppo-checkpoint-path <path-to-policy-checkpoint> \
--ddppo-config-path config_files/ddppo/ddppo_pointnav_2021.yaml \
--vo-config-path <path-to-vo-config> \
--vo-checkpoint-path <path-to-vo-checkpoint> \
--pth-gpu-id 0 \
--rotation-regularization-on \
--vertical-flip-on
```

## Pre-trained Models
Checkpoints may be downloaded from Google Drive manually or by using gdown.

### Navigation Policy
<table> 
<caption></caption>
<tr>
    <th>Training scenes</th> 
    <th>Terminal reward</th> 
    <th>Download</th>
    <th>Task setting</th>
</tr>
<tr>
    <td>Gibson 4+</td> 
    <td>2.5 Success</td> 
    <td><a href="">Link</a></td>
    <td>Habitat Challenge 2021</td>
</tr>
<tr>
    <td>Gibson 0+</td> 
    <td>2.5 SPL</td> 
    <td><a href="">Link</a></td>
    <td>Habitat Challenge 2021</td>
</tr> 
<tr>
    <td>HM3D-MP3D-Gibson 0+</td> 
    <td>2.5 SPL</td> 
    <td><a href="">Link</a></td>
    <td>Sim2real</td>
</tr> 
</table>

To see the policy training config, download the checkpoint and execute command below:
```python
import torch
checkpoint = torch.load('path-to-the-policy-checkpoint')
print(checkpoint['config'])
```

### Visual Odometry

<table> 
<caption></caption> 
<tr> 
    <th rowspan="2"></th>
    <th rowspan="2">Dataset Size(M)</th> 
    <th colspan="2">VO</th> 
    <th colspan="2">Embedding</th> 
    <th colspan="2">Train time</th>
    <th rowspan="2">Epoch</th>
    <th rowspan="2">Download</th>
    <th rowspan="2" style="visibility: hidden"></th>
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
    <td><a href="https://drive.google.com/drive/folders/1Mnv1ZXxlH-3S7F7TEW8OXo7heh0ZruwY?usp=sharing">Link</a></td>
    <td rowspan="10">Gibson</td>
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
    <td><a href="https://drive.google.com/drive/folders/1rI_TtBvpe24U3p-hxl8Ccg1SxI0VAQcZ?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1UDyLzxUkjnpY3knHHW5vWjVeScwM3yYj?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1ujKBuhMFI9pv8ji5I1Iz_vWccU-dNRjB?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1M_LyxGE-qFpRhgRzDunB8Elnf4ElY4aW?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1KURvdGKb7CeZ_Y4U-69wkmjVFOmXCcdh?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1LflyaYl1Vjpkhhl3kZkKEiZnEcKa4Vq0?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1Nle9EFgi-RAjkwSZ5j-cEzuxOA21XolB?usp=sharing">Link</a></td>
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
    <td><a href="https://drive.google.com/drive/folders/1ja9Dx8fnzv1lypvNMKqUrloDjbHLvcDO?usp=sharing">Link</a></td>
</tr> 
<tr> 
    <td>10</td> 
    <td>5</td> 
    <td>ResNet50</td> 
    <td>7.6</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>64</td>
    <td><a href="https://drive.google.com/drive/folders/14PYwFKum8m1CYdrt9IyOWj6wlFFyr0Ly?usp=sharing">Link</a></td>
</tr> 

<tr> 
    <td>11</td> 
    <td>?</td> 
    <td>ResNet50</td> 
    <td>7.6</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>32</td>
    <td><a href="https://drive.google.com/drive/folders/193Ih1w13fdbimB8IO1fgykF4k9Fkk8h1?usp=sharing">Link</a></td>
    <td rowspan="1">MP3D fine-tuned</td>
</tr> 

<tr> 
    <td>12</td> 
    <td>?</td> 
    <td>ResNet50</td> 
    <td>7.6</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>&#10004;</td> 
    <td>56</td>
    <td><a href="https://drive.google.com/drive/folders/12XUM5Fd0ru6cWb_PAm3rIvsvDwK75mnD?usp=sharing">Link</a></td>
    <td rowspan="1">Sim2real</td>
</tr> 
</table>

## Results
We improve Realistic PointNav agent navigation performance from 64% Success / 52% SPL to 96% Success and 77% SPL on
the Gibson val split, and achieve the following performance on:

<table> 
<caption>

[Habitat Challenge 2021](https://eval.ai/web/challenges/challenge-page/802/leaderboard/2192) benchmark test-standard split (retrieved 2021-Nov-16).

</caption> 
<tr> 
    <th>Rank</th>
    <th>Participant team</th> 
    <th>SPL</th> 
    <th>SoftSPL</th> 
    <th>Distance to goal</th> 
    <th>Success</th>
</tr> 
<tr> 
    <td>1</td> 
    <td>VO for Realistic PointGoal (Ours)</td> 
    <td>0.74</td> 
    <td>0.76</td> 
    <td>0.21</td> 
    <td>0.94</td>
</tr> 
<tr> 
    <td>2</td> 
    <td>inspir.ai robotics</td> 
    <td>0.70</td> 
    <td>0.71</td> 
    <td>0.70</td> 
    <td>0.91</td>
</tr> 
<tr> 
    <td>3</td> 
    <td>VO2021</td> 
    <td>0.59</td> 
    <td>0.69</td> 
    <td>0.53</td> 
    <td>0.78</td>
</tr> 
<tr> 
    <td>4</td> 
    <td>Differentiable SLAM-net</td> 
    <td>0.47</td> 
    <td>0.60</td> 
    <td>1.74</td> 
    <td>0.65</td>
</tr> 
</table>

We have deployed our agent (with no sim2real adaptation) onto a LoCoBot. It achieves 11%Success, 71%SoftSPL, 
and makes it 92% of the way to the goal (SoftSuccess). See 3rd-person videos and mapped routes
on our [website](https://rpartsey.github.io/pointgoalnav/). 


## Citing
If you use [IMN-RPG](https://github.com/rpartsey/pointgoal-navigation) in your research, please cite our paper:
```tex
@InProceedings{Partsey_2022_CVPR,
    author    = {Partsey, Ruslan and Wijmans, Erik and Yokoyama, Naoki and Dobosevych, Oles and Batra, Dhruv and Maksymets, Oleksandr},
    title     = {Is Mapping Necessary for Realistic PointGoal Navigation?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {17232-17241}
}
```
