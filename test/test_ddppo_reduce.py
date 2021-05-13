#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest

from habitat.core.spaces import ActionSpace, EmptySpace
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor

torch = pytest.importorskip("torch")
habitat_baselines = pytest.importorskip("habitat_baselines")

import gym
from torch import distributed as distrib
from torch import nn

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy


from odometry.config.default import get_config as get_odometry_config
from odometry.models import make_model
from odometry.losses import make_loss
from odometry.models.models import init_distributed


def _worker_fn(
    world_rank: int, world_size: int, port: int, unused_params: bool
):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    tcp_store = distrib.TCPStore(  # type: ignore
        "127.0.0.1", port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        "gloo", store=tcp_store, rank=world_rank, world_size=world_size
    )

    config = get_config("test/config_files/ppo_pointnav_test.yaml")
    obs_space = gym.spaces.Dict(
        {
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid: gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }
    )
    action_space = ActionSpace({"move": EmptySpace()})
    actor_critic = PointNavBaselinePolicy.from_config(
        config, obs_space, action_space
    )
    # This use adds some arbitrary parameters that aren't part of the computation
    # graph, so they will mess up DDP if they aren't correctly ignored by it
    if unused_params:
        actor_critic.unused = nn.Linear(64, 64)

    actor_critic.to(device=device)
    ppo_cfg = config.RL.PPO
    agent = DDPPO(
        actor_critic=actor_critic,
        clip_param=ppo_cfg.clip_param,
        ppo_epoch=ppo_cfg.ppo_epoch,
        num_mini_batch=ppo_cfg.num_mini_batch,
        value_loss_coef=ppo_cfg.value_loss_coef,
        entropy_coef=ppo_cfg.entropy_coef,
        lr=ppo_cfg.lr,
        eps=ppo_cfg.eps,
        max_grad_norm=ppo_cfg.max_grad_norm,
        use_normalized_advantage=ppo_cfg.use_normalized_advantage,
    )
    agent.init_distributed()
    rollouts = RolloutStorage(
        ppo_cfg.num_steps,
        2,
        obs_space,
        action_space,
        ppo_cfg.hidden_size,
        num_recurrent_layers=actor_critic.net.num_recurrent_layers,
        is_double_buffered=False,
    )
    rollouts.to(device)

    for k, v in rollouts.buffers["observations"].items():
        rollouts.buffers["observations"][k] = torch.randn_like(v)

    # Add two steps so batching works
    rollouts.advance_rollout()
    rollouts.advance_rollout()

    # Get a single batch
    batch = next(rollouts.recurrent_generator(rollouts.buffers["returns"], 1))

    # Call eval actions through the internal wrapper that is used in
    # agent.update
    value, action_log_probs, dist_entropy, _ = agent._evaluate_actions(
        batch["observations"],
        batch["recurrent_hidden_states"],
        batch["prev_actions"],
        batch["masks"],
        batch["actions"],
    )
    # Backprop on things
    (value.mean() + action_log_probs.mean() + dist_entropy.mean()).backward()

    # Make sure all ranks have very similar parameters
    for param in actor_critic.parameters():
        if param.grad is not None:
            grads = [param.grad.detach().clone() for _ in range(world_size)]
            distrib.all_gather(grads, grads[world_rank])

            for i in range(world_size):
                assert torch.isclose(grads[i], grads[world_rank]).all()


    config_path = 'config_files/odometry/resnet18_bs16_ddepth5_maxd0.5_randomsampling_dropout0.15_poseloss1._1._180x320_embedd_act_hc2021_vo2_joint.yaml'
    config = get_odometry_config(config_path, new_keys_allowed=True)

    # config.defrost()
    # config.experiment_dir = os.path.join(config.log_dir, config.experiment_name)
    # config.tb_dir = os.path.join(config.experiment_dir, 'tb')
    # config.model.best_checkpoint_path = os.path.join(config.experiment_dir, 'best_checkpoint.pt')
    # config.model.last_checkpoint_path = os.path.join(config.experiment_dir, 'last_checkpoint.pt')
    # config.config_save_path = os.path.join(config.experiment_dir, 'config.yaml')
    # config.freeze()

    # init_experiment(config)
    # set_random_seed(config.seed) TODO: check if the seed can be set here

    vo_model = make_model(config.model)
    if unused_params:
        vo_model.unused = nn.Linear(64, 64)
    vo_model.to(device)
    vo_model = init_distributed(vo_model, device, find_unused_params=True)
    vo_loss_f = make_loss(config.loss)

    encoder_params = config.model.params.encoder.params
    batch_size = config.train.loader.params.batch_size
    in_channels = encoder_params.in_channels
    in_height = encoder_params.in_height
    in_width = encoder_params.in_width

    target = torch.rand(batch_size, 4).to(device)
    batch = torch.rand(batch_size, in_channels, in_height, in_width).to(device)
    print(batch[2,4, 24:28, 97:100])
    output = vo_model(batch, **{})
    vo_loss, vo_loss_components = vo_loss_f(output, target)
    vo_loss.backward()

    # Make sure all ranks have very similar parameters
    for param in vo_model.parameters():
        if param.grad is not None:
            grads = [param.grad.detach().clone() for _ in range(world_size)]
            distrib.all_gather(grads, grads[world_rank])

            for i in range(world_size):
                assert torch.isclose(grads[i], grads[world_rank]).all()


@pytest.mark.parametrize("unused_params", [True, False])
def test_ddppo_reduce(unused_params: bool):
    world_size = 2
    torch.multiprocessing.spawn(
        _worker_fn,
        args=(world_size, 8748 + int(unused_params), unused_params),
        nprocs=world_size,
    )
