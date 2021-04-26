import contextlib
import os
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
from gym import spaces
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_interrupted_state,
    rank0_only,
    requeue_job,
    save_interrupted_state,
)
from habitat_baselines.utils.common import batch_obs

# from .ppo_joint import PPO, DDPPO
from odometry.config.default import get_config
from odometry.dataset import make_transforms
from odometry.models import make_model
from odometry.losses import make_loss
from odometry.models.models import init_distributed
from odometry.optims import make_optimizer
from odometry.utils import transform_batch


@baseline_registry.register_trainer(name="ddppo-joint")
@baseline_registry.register_trainer(name="ppo-joint")
class PPOTrainerJoint(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.vo_device = None
        self.vo_batch_size = None
        self.vo_model = None
        self.observations_transforms = None
        self.vo_optimizer = None
        self.vo_loss_f = None
        self.depth_discretization_on = None
        self.num_updates_done = 0
        self.vo_updates_counter = 0

    def _setup_visual_odometry(self, device_id):
        # VO model initialization
        config_path = 'config_files/odometry/resnet18_bs16_ddepth5_maxd0.5_randomsampling_dropout0.15_poseloss1._1._180x320_embedd_act_hc2021_vo2_joint.yaml'
        config = get_config(config_path, new_keys_allowed=True)

        # config.defrost()
        # config.experiment_dir = os.path.join(config.log_dir, config.experiment_name)
        # config.tb_dir = os.path.join(config.experiment_dir, 'tb')
        # config.model.best_checkpoint_path = os.path.join(config.experiment_dir, 'best_checkpoint.pt')
        # config.model.last_checkpoint_path = os.path.join(config.experiment_dir, 'last_checkpoint.pt')
        # config.config_save_path = os.path.join(config.experiment_dir, 'config.yaml')
        # config.freeze()

        # init_experiment(config)
        # set_random_seed(config.seed) TODO: check if the seed can be set here

        self.vo_device = torch.device("cuda", device_id)
        self.vo_batch_size = config.train.loader.params.batch_size
        self.vo_model = make_model(config.model).to(self.vo_device)
        self.observations_transforms = make_transforms(config.train.dataset.transforms)
        self.vo_optimizer = make_optimizer(config.optim, self.vo_model.parameters())
        self.vo_loss_f = make_loss(config.loss)
        self.depth_discretization_on = config.val.dataset.transforms.DiscretizeDepth.params.n_channels > 0
        # self.num_updates_done = 0
        self.vo_updates_counter = 0

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        # pop rgb sensor
        rgb_sensor_space = observation_space.spaces.pop('rgb')

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )

        # add rgb sensor back
        # observation_space.spaces['rgb'] = rgb_sensor_space

        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
                self.config.RL.DDPPO.pretrained_encoder
                or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
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

    def _init_train(self):
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_world_size()
                * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1
        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        # pop rgb sensor
        batch.pop('rgb')

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        # Visual odometry setup
        self._setup_visual_odometry(local_rank)
        if self._is_distributed:
            self.vo_model = init_distributed(self.vo_model, self.vo_device, find_unused_params=True)

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    def _train_visual_odometry(self):
        # Visual Odometry
        observations = self.rollouts.buffers["observations"]
        dones = self.rollouts.buffers["masks"].clone().detach()

        # sampling frames
        T = self.rollouts.numsteps + 1
        N = dones.view(T, -1).size(1)

        rollout_boundaries = dones
        rollout_boundaries = torch.logical_not(rollout_boundaries).nonzero(as_tuple=False)

        episode_starts_transposed = (rollout_boundaries[:, 1] * T + rollout_boundaries[:, 0])
        episode_starts_transposed = torch.cat([
            episode_starts_transposed,
            (torch.arange(1, N+1) * T).to(episode_starts_transposed.device)
        ])
        episode_starts_transposed, sorted_indices = torch.sort(
            episode_starts_transposed, descending=False
        )
        rollout_intervals = list(zip(
            episode_starts_transposed[:-1].cpu().numpy().tolist(),
            episode_starts_transposed[1:].cpu().numpy().tolist()
        ))

        # assert len(rollout_intervals) > 0, f'\nDones:\n{dones}\nrollout_boundaries:\n{rollout_boundaries}\nepisode_starts_transposed:\n{episode_starts_transposed}\nrollout_intervals:\n{rollout_intervals}\n'

        rollout_intervals = list(filter(lambda interval: interval[1] - interval[0] > 1, rollout_intervals))
        rollout_intervals = sorted(rollout_intervals, key=lambda interval: interval[1] - interval[0], reverse=True)

        # assert len(rollout_intervals) > 0, f'\nDones:\n{dones}\nrollout_boundaries:\n{rollout_boundaries}\nepisode_starts_transposed:\n{episode_starts_transposed}\nrollout_intervals:\n{rollout_intervals}\n'

        if sum([interv[1] - interv[0] for interv in rollout_intervals]) >= self.vo_batch_size:
            source_frame_indices = []
            for interval in rollout_intervals:
                source_frame_indices.extend(np.arange(*interval)[:-1])

            num_frames_to_sample = (len(source_frame_indices) // self.vo_batch_size) * self.vo_batch_size
            num_epochs = 2

            self.vo_model.train()
            vo_metrics = defaultdict(lambda: 0)
            for e in range(num_epochs):
                np.random.shuffle(source_frame_indices)
                for batch_i in range(0, num_frames_to_sample, self.vo_batch_size):
                    batch_indices = source_frame_indices[batch_i:batch_i+self.vo_batch_size]

                    egomotions = []
                    visual_observations = []
                    for index in batch_indices:
                        i = index % T
                        j = index // T
                        # assert i != 128, f'i != 128 {index}'
                        visual_observations.append({
                            'source_rgb': observations['vo_rgb'][i, j],  # .cpu().numpy(),
                            'target_rgb': observations['vo_rgb'][i + 1, j],  # .cpu().numpy(),
                            'source_depth': observations['vo_depth'][i, j],  # .cpu().numpy(),
                            'target_depth': observations['vo_depth'][i + 1, j]  # .cpu().numpy()
                        })
                        egomotions.append(observations['egomotion'][i + 1, j])

                    visual_observations = [
                        self.observations_transforms(obs)
                        for obs in visual_observations
                    ]
                    batch = {
                        'source_rgb': [],
                        'target_rgb': [],
                        'source_depth': [],
                        'target_depth': [],
                        'source_depth_discretized': [],
                        'target_depth_discretized': []
                    }
                    for obs in visual_observations:
                        batch['source_rgb'].append(obs['source_rgb'])
                        batch['target_rgb'].append(obs['target_rgb'])
                        batch['source_depth'].append(obs['source_depth'])
                        batch['target_depth'].append(obs['target_depth'])
                        batch['source_depth_discretized'].append(obs['source_depth_discretized'])
                        batch['target_depth_discretized'].append(obs['source_depth_discretized'])

                    batch['source_rgb'] = torch.stack(batch['source_rgb'])
                    batch['target_rgb'] = torch.stack(batch['target_rgb'])
                    batch['source_depth'] = torch.stack(batch['source_depth'])
                    batch['target_depth'] = torch.stack(batch['target_depth'])
                    batch['source_depth_discretized'] = torch.stack(batch['source_depth_discretized'])
                    batch['target_depth_discretized'] = torch.stack(batch['target_depth_discretized'])

                    batch, embeddings, _ = transform_batch(batch)
                    batch = batch.to(self.vo_device)
                    for k, v in embeddings.items():
                        embeddings[k] = v.to(self.vo_device)

                    output = self.vo_model(batch, **embeddings)
                    target = torch.stack(egomotions).to(self.vo_device)
                    vo_loss, vo_loss_components = self.vo_loss_f(output, target)

                    self.vo_optimizer.zero_grad()
                    vo_loss.backward()
                    self.vo_optimizer.step()

                    vo_metrics['loss'] += vo_loss.item()
                    for loss_component, value in vo_loss_components.items():
                        vo_metrics[loss_component] += value.item()

                    self.vo_updates_counter += 1

            if self.vo_updates_counter:
                for metric_name in vo_metrics:
                    vo_metrics[metric_name] /= self.vo_updates_counter
                self.vo_updates_counter = 0

            return vo_metrics

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]

        ppo_cfg = self.config.RL.PPO

        with (
                SummaryWriter(log_dir='vo_tb')
                if rank0_only()
                else contextlib.suppress()
        ) as vo_writer:
            with (
                    TensorboardWriter(
                        self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
                    )
                    if rank0_only()
                    else contextlib.suppress()
            ) as writer:
                while not self.is_done():
                    profiling_wrapper.on_start_step()
                    profiling_wrapper.range_push("train update")

                    if ppo_cfg.use_linear_clip_decay:
                        self.agent.clip_param = ppo_cfg.clip_param * (
                                1 - self.percent_done()
                        )

                    if EXIT.is_set():
                        profiling_wrapper.range_pop()  # train update

                        self.envs.close()

                        if REQUEUE.is_set() and rank0_only():
                            requeue_stats = dict(
                                env_time=self.env_time,
                                pth_time=self.pth_time,
                                count_checkpoints=count_checkpoints,
                                num_steps_done=self.num_steps_done,
                                num_updates_done=self.num_updates_done,
                                _last_checkpoint_percent=self._last_checkpoint_percent,
                                prev_time=(time.time() - self.t_start) + prev_time,
                            )
                            save_interrupted_state(
                                dict(
                                    state_dict=self.agent.state_dict(),
                                    optim_state=self.agent.optimizer.state_dict(),
                                    lr_sched_state=lr_scheduler.state_dict(),
                                    config=self.config,
                                    requeue_stats=requeue_stats,
                                )
                            )

                        requeue_job()
                        return

                    self.agent.eval()
                    count_steps_delta = 0
                    profiling_wrapper.range_push("rollouts loop")

                    profiling_wrapper.range_push("_collect_rollout_step")
                    for buffer_index in range(self._nbuffers):
                        self._compute_actions_and_step_envs(buffer_index)

                    for step in range(ppo_cfg.num_steps):
                        is_last_step = (
                                self.should_end_early(step + 1)
                                or (step + 1) == ppo_cfg.num_steps
                        )

                        for buffer_index in range(self._nbuffers):
                            count_steps_delta += self._collect_environment_result(
                                buffer_index
                            )

                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_pop()  # _collect_rollout_step

                            if not is_last_step:
                                if (buffer_index + 1) == self._nbuffers:
                                    profiling_wrapper.range_push(
                                        "_collect_rollout_step"
                                    )

                                self._compute_actions_and_step_envs(buffer_index)

                        if is_last_step:
                            break

                    profiling_wrapper.range_pop()  # rollouts loop

                    if self._is_distributed:
                        self.num_rollouts_done_store.add("num_done", 1)

                    # Visual odometry update
                    vo_metrics = self._train_visual_odometry()

                    (
                        value_loss,
                        action_loss,
                        dist_entropy
                    ) = self._update_agent()

                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler.step()  # type: ignore

                    self.num_updates_done += 1
                    losses = self._coalesce_post_step(
                        dict(value_loss=value_loss, action_loss=action_loss),
                        count_steps_delta,
                    )

                    self._training_log(writer, losses, prev_time)
                    if vo_writer:
                        for k, v in vo_metrics.items():
                            vo_writer.add_scalar(f'metrics/{k}', v, self.num_updates_done)

                    # checkpoint model
                    if rank0_only() and self.should_checkpoint():
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(
                                step=self.num_steps_done,
                                wall_time=(time.time() - self.t_start) + prev_time,
                            ),
                        )
                        count_checkpoints += 1
                        torch.save(self.vo_model.state_dict(), f'vo_tb/ckpt_{self.num_updates_done}.pt')

                    profiling_wrapper.range_pop()  # train update

                self.envs.close()
