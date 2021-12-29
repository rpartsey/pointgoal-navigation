# DD-PPO fine-tuning
DD-PPO fine-tuning using trained VO as a replacement of GPS+Compass sensor.

**Context:** the VO is wrapped in `habitat_extensions.sensors.EgomotionPointGoalSensor` that is used 
as a replacement of `IntegratedPointGoalGPSAndCompassSensor` and run as part of the habitat simulator.

**Concerns:** the VO model is run on GPU (`habitat_config.HABITAT_SIM_V0.GPU_DEVICE_ID`) thus the total number of DD-PPO 
parallel processes (`NUM_PROCESSES`) that can fit on one GPU is 2-4 times smaller than DD-PPO trained with GT localization.

### Best pretrained checkpoints
Download the best checkpoints if needed:
- [DD-PPO](https://drive.google.com/file/d/1D-ZMg68grmlj-o7kiDegU_dokOkguh8Z/view?usp=sharing)
- [VO](https://drive.google.com/file/d/1HT1npoOG_X6atO8BzaXexbnnWOpWQW_m/view?usp=sharing)
  - [config](https://drive.google.com/file/d/1Dd6Ldu2drUS-UT4207mwnkWHSHFkXWji/view?usp=sharing)

### DD-PPO config
Use `config_files/ddppo/ddppo_pointnav_2021_finetuning.yaml` as an example DD-PPO fine-tuning config. 
Set `RL.DDPPO.pretrained_weights` to point to pretrained DD-PPO checkpoint.

Try different values (True/False) for parameters:
```yaml
    # Initialize just the visual encoder backbone with pretrained weights
    pretrained_encoder: False
    # Whether or not the visual encoder backbone will be trained.
    train_encoder: True
    # Whether or not to reset the critic linear layer
    reset_critic: False
```
Earlier we run smoke fine-tuning experiments with `train_encoder` set to `True` and `False`, 
but our experiments improve neither Success nor SPL (of not fine-tuned DD-PP + VO). 

### Base task config
Use `config_files/challenge_pointnav2021_egomotion_pointgoal_sensor.local.rgbd.yaml` as an example base task config.
Set `TASK.EGOMOTION_POINTGOAL_SENSOR.TRAIN_CONFIG_PATH` and `TASK.EGOMOTION_POINTGOAL_SENSOR.CHECKPOINT_PATH` to point 
to the best VO config and checkpoint respectively.

### Run command
The fine-tuning can be run the same way GT DD-PPO is trained. See `navigation/experiments/experiment_launcher.sh` and `navigation/experiments/run_experiment.sh`.

Replace
```
python -u  run_ddppo.py \
    --run-type train ${CMD_OPTS}
```

with 

```shell
python -u run_customized_ddppo.py \
 --run-type train \
 --exp-config config_files/ddppo/ddppo_pointnav_2021_finetuning.yaml \
 NUM_PROCESSES 4 \
 BASE_TASK_CONFIG_PATH config_files/challenge_pointnav2021_egomotion_pointgoal_sensor.local.rgbd.yaml \
 ${CMD_OPTS}
```

(Pick `NUM_PROCESSES` number to fit GPU memory)