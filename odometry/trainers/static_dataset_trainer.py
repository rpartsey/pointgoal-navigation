from collections import defaultdict
from datetime import datetime

from tqdm import tqdm
from habitat_baselines.rl.ddppo.algo.ddp_utils import rank0_only

from .base_trainer import BaseTrainer
from ..utils import transform_batch
from ..metrics import action_id_to_action_name


class StaticDatasetTrainer(BaseTrainer):
    def update_distrib_config(self, local_rank):
        self.config.defrost()
        self.config.device = local_rank
        self.config.train.loader.is_distributed = True
        self.config.val.loader.is_distributed = True
        self.config.freeze()

    def train_epoch(self):
        self.model.train()

        num_items = 0
        num_items_per_action = defaultdict(lambda: 0)

        metrics = defaultdict(lambda: 0)

        for data in tqdm(self.train_loader, disable=self.is_distributed()):
            data, embeddings, target = transform_batch(data)
            data = data.float().to(self.device)
            target = target.float().to(self.device)
            for k, v in embeddings.items():
                embeddings[k] = v.to(self.device)

            output = self.model(data, **embeddings)
            loss, loss_components = self.loss_f(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = target.shape[0]
            metrics['loss'] += loss.item() * batch_size
            for loss_component, value in loss_components.items():
                metrics[loss_component] += value.item() * batch_size
            for metric_f in self.train_metric_fns:
                metrics[metric_f.__name__] += metric_f(output, target).item() * batch_size
                if self.config.compute_metrics_per_action:
                    for action_id in embeddings['action'].unique():
                        action_name = action_id_to_action_name[action_id.item()]
                        action_mask = embeddings['action'] == action_id
                        action_metric_name = f'{metric_f.__name__}_{action_name}'
                        num_action_items = action_mask.sum()

                        metrics[action_metric_name] += metric_f(output[action_mask], target[action_mask]).item() * num_action_items
                        num_items_per_action[action_metric_name] += num_action_items

            num_items += batch_size

        for metric_name in metrics:
            metrics[metric_name] /= num_items_per_action.get(metric_name, num_items)

        return metrics

    def val_epoch(self):
        pass

    def loop(self):
        # for epoch in range(1, self.config.epochs + 1):
        #     if rank0_only():
        #         print(f'{datetime.now()} Epoch {epoch}')
        #
        #     train_metrics = train(model, optimizer, train_loader, loss_f, train_metric_fns, device, is_distributed,
        #                           config.compute_metrics_per_action)
        #     if is_distributed:
        #         train_metrics = coalesce_post_step(train_metrics, device)
        #     write_metrics(epoch, train_metrics, train_writer)
        #     print_metrics('Train', train_metrics)
        #
        #     val_metrics = val(model, val_loader, loss_f, val_metric_fns, device, is_distributed,
        #                       config.compute_metrics_per_action)
        #     if is_distributed:
        #         val_metrics = coalesce_post_step(val_metrics, device)
        #     write_metrics(epoch, val_metrics, val_writer)
        #     print_metrics('Val', val_metrics)
        #
        #     if hasattr(config, 'train_val'):
        #         train_val_metrics = val(model, train_val_loader, loss_f, train_val_metric_fns, device, is_distributed,
        #                                 config.compute_metrics_per_action)
        #         if is_distributed:
        #             val_metrics = coalesce_post_step(val_metrics, device)
        #         write_metrics(epoch, train_val_metrics, train_val_writer)
        #         print_metrics('Train-val', train_val_metrics)
        #
        #     early_stopping(val_metrics['loss'])
        #     if rank0_only() and config.model.save:  # and early_stopping.counter == 0:
        #         best_checkpoint_path = config.model.best_checkpoint_path.replace('.pt', f'_{str(epoch).zfill(3)}e.pt')
        #         torch.save(model.state_dict(), best_checkpoint_path)
        #         print('Saved best model checkpoint to disk.')
        #     if early_stopping.early_stop:
        #         print(f'Early stopping after {epoch} epochs.')
        #         break
        #
        #     if scheduler:
        #         scheduler.step()
        pass
