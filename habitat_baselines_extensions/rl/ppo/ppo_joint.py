from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch

from habitat.utils import profiling_wrapper
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.rl.ppo import PPO as BaselinePPO
from habitat_baselines.rl.ddppo.algo.ddppo import DecentralizedDistributedMixin

from odometry.config.default import get_config
from odometry.dataset import make_transforms
from odometry.models import make_model
from odometry.losses import make_loss
from odometry.optims import make_optimizer
from train_odometry import transform_batch


class PPO(BaselinePPO):
    def __init__(
            self,
            actor_critic: Policy,
            clip_param: float,
            ppo_epoch: int,
            num_mini_batch: int,
            value_loss_coef: float,
            entropy_coef: float,
            lr: Optional[float] = None,
            eps: Optional[float] = None,
            max_grad_norm: Optional[float] = None,
            use_clipped_value_loss: bool = True,
            use_normalized_advantage: bool = True,
            vo_writer=None
    ) -> None:
        super().__init__(
            actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
            use_clipped_value_loss,
            use_normalized_advantage
        )
        # VO model initialization
        config_path = 'config_files/odometry/vo_net.yaml'
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

        self.vo_device = torch.device(config.device)
        self.vo_model_batch_size = config.train.loader.params.batch_size
        self.vo_model = make_model(config.model).to(self.vo_device)
        self.observations_transforms = make_transforms(config.train.dataset.transforms)
        self.vo_optimizer = make_optimizer(config.optim, self.vo_model.parameters())
        self.vo_loss_f = make_loss(config.loss)
        self.vo_writer = vo_writer
        self.num_updates_done = 0
        self.vo_updates_counter = 0

    def update(self, rollouts: RolloutStorage) -> Tuple[float, float, float]:
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0
        vo_metrics = defaultdict(lambda: 0)

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for batch in data_generator:
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                )

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * batch["advantages"]
                )
                action_loss = -(torch.min(surr1, surr2).mean())

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - batch["returns"]
                    ).pow(2)
                    value_loss = 0.5 * torch.max(
                        value_losses, value_losses_clipped
                    )
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss = value_loss.mean()
                dist_entropy = dist_entropy.mean()

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

                # Visual Odometry
                observations = batch["observations"]
                dones = batch["masks"].clone().detach()
                batch_len = len(observations['depth'])

                # sampling frames
                T = rollouts.numsteps
                N = dones.view(T, -1).size(1)

                rollout_boundaries = dones
                rollout_boundaries[0] = True
                rollout_boundaries = rollout_boundaries.nonzero(as_tuple=False)

                episode_starts_transposed = (rollout_boundaries[:, 1] * T + rollout_boundaries[:, 0])
                episode_starts_transposed = torch.cat([
                    episode_starts_transposed,
                    torch.tensor(T * N).unsqueeze(0).to(episode_starts_transposed.device)
                ])
                episode_starts_transposed, sorted_indices = torch.sort(
                    episode_starts_transposed, descending=False
                )
                rollout_intervals = list(zip(
                    episode_starts_transposed[:-1].cpu().numpy().tolist(),
                    episode_starts_transposed[1:].cpu().numpy().tolist()
                ))

                # assert len(rollout_intervals) > 0, f'\nDones:\n{dones}\nrollout_boundaries:\n{rollout_boundaries}\nepisode_starts_transposed:\n{episode_starts_transposed}\nrollout_intervals:\n{rollout_intervals}\n'

                rollout_intervals = filter(lambda interval: interval[1] - interval[0] > 1, rollout_intervals)
                rollout_intervals = sorted(rollout_intervals, key=lambda interval: interval[1] - interval[0],
                                           reverse=True)

                # assert len(rollout_intervals) > 0, f'\nDones:\n{dones}\nrollout_boundaries:\n{rollout_boundaries}\nepisode_starts_transposed:\n{episode_starts_transposed}\nrollout_intervals:\n{rollout_intervals}\n'

                if sum([interv[1] - interv[0] for interv in rollout_intervals]) >= self.vo_model_batch_size:
                    max_interval_len = rollout_intervals[0][1] - rollout_intervals[0][0]
                    source_frame_indices = np.full((max_interval_len, len(rollout_intervals)), -1)
                    for i in range(len(rollout_intervals)):
                        rollout_len = rollout_intervals[i][1] - rollout_intervals[i][0]
                        rollout_frame_indices = np.arange(*rollout_intervals[i])[:-1]
                        np.random.shuffle(rollout_frame_indices)

                        source_frame_indices[:(rollout_len-1), i] = rollout_frame_indices

                    source_frame_indices_flat = source_frame_indices.flatten()
                    frame_indices = source_frame_indices_flat[np.where(source_frame_indices_flat > -1)]

                    egomotions = []
                    visual_observations = []
                    for i in frame_indices[:self.vo_model_batch_size]:
                        visual_observations.append({
                            'source_rgb': observations['rgb'][i].cpu().numpy(),
                            'target_rgb': observations['rgb'][i+1].cpu().numpy(),
                            'source_depth': observations['depth'][i].cpu().numpy(),
                            'target_depth': observations['depth'][i+1].cpu().numpy()
                        })
                        egomotions.append(observations['egomotion'][i+1])

                    visual_observations = [
                        self.observations_transforms(obs)
                        for obs in visual_observations
                    ]
                    batch = {
                        'source_rgb': [],
                        'target_rgb': [],
                        'source_depth': [],
                        'target_depth': [],
                    }
                    for obs in visual_observations:
                        batch['source_rgb'].append(obs['source_rgb'])
                        batch['target_rgb'].append(obs['target_rgb'])
                        batch['source_depth'].append(obs['source_depth'])
                        batch['target_depth'].append(obs['target_depth'])

                    batch['source_rgb'] = torch.stack(batch['source_rgb'])
                    batch['target_rgb'] = torch.stack(batch['target_rgb'])
                    batch['source_depth'] = torch.stack(batch['source_depth'])
                    batch['target_depth'] = torch.stack(batch['target_depth'])

                    batch, _ = transform_batch(batch)
                    batch = batch.to(self.vo_device)

                    self.vo_model.train()
                    output = self.vo_model(batch)
                    target = torch.stack(egomotions).to(self.vo_device)
                    vo_loss, vo_loss_components = self.vo_loss_f(output, target)

                    self.vo_optimizer.zero_grad()
                    vo_loss.backward()
                    self.vo_optimizer.step()

                    vo_metrics['loss'] += vo_loss.item()
                    for loss_component, value in vo_loss_components.items():
                        vo_metrics[loss_component] += value.item()

                    self.vo_updates_counter += 1

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        for metric_name in vo_metrics:
            vo_metrics[metric_name] /= num_updates

        if self.vo_updates_counter:
            self.num_updates_done += 1
            for k, v in vo_metrics.items():
                self.vo_writer.add_scalar(f'metrics/{k}', v, self.num_updates_done)

            self.vo_updates_counter = 0

        if self.num_updates_done % 4000 == 0:
            torch.save(self.vo_model.state_dict(), f'vo_tb/ckpt_{self.num_updates_done}.pt')

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
