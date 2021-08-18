import os
import shutil
import argparse
from collections import defaultdict

import numpy as np
from habitat_baselines.rl.ddppo.algo.ddp_utils import get_distrib_size, init_distrib_slurm, rank0_only
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from odometry.models.models import init_distributed
from odometry.config.default import get_config
from odometry.models import make_model
from odometry.dataset import make_dataset, make_data_loader
from odometry.losses import make_loss
from odometry.optims import make_optimizer
from odometry.metrics import make_metrics, action_id_to_action_name
from odometry.utils import set_random_seed, transform_batch


@rank0_only
def print_metrics(phase, metrics):

    metrics_log_str = ' '.join([
        '\t{}: {:.6f}\n'.format(k, v)
        for k, v in metrics.items()
    ])

    print(f'{phase}:\n {metrics_log_str}')


@rank0_only
def write_metrics(epoch, metrics, writer):
    for metric_name, value in metrics.items():
        key = 'losses' if 'loss' in metric_name else 'metrics'
        writer.add_scalar(f'{key}/{metric_name}', value, epoch)


@rank0_only
def init_experiment(config):
    if os.path.exists(config.experiment_dir):
        def ask():
            return input(f'Experiment "{config.experiment_name}" already exists. Delete (y/n)?')

        answer = ask()
        while answer not in ('y', 'n'):
            answer = ask()

        delete = answer == 'y'
        if not delete:
            exit(1)

        shutil.rmtree(config.experiment_dir)

    os.makedirs(config.experiment_dir)
    with open(config.config_save_path, 'w') as dest_file:
        config.dump(stream=dest_file)


def _all_reduce(t: torch.Tensor, device) -> torch.Tensor:
    orig_device = t.device
    t = t.to(device)
    torch.distributed.all_reduce(t)

    return t.to(orig_device)


def coalesce_post_step(metrics, device):
    metric_name_ordering = sorted(metrics.keys())
    stats = torch.tensor(
        [metrics[k] for k in metric_name_ordering],
        device="cpu",
        dtype=torch.float32,
    )
    stats = _all_reduce(stats, device)
    stats /= torch.distributed.get_world_size()

    return {
        k: stats[i].item() for i, k in enumerate(metric_name_ordering)
    }


def train(model, optimizer, train_loader, loss_f, metric_fns, device, disable_tqdm=False, compute_metrics_per_action=True):
    model.train()

    num_items = 0
    num_items_per_action = defaultdict(lambda: 0)

    metrics = defaultdict(lambda: 0)

    for data in tqdm(train_loader, disable=disable_tqdm):
        data, embeddings, target = transform_batch(data)
        data = data.float().to(device)
        target = target.float().to(device)
        for k, v in embeddings.items():
            embeddings[k] = v.to(device)

        output = model(data, **embeddings)
        loss, loss_components = loss_f(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        metrics['loss'] += loss.item() * batch_size
        for loss_component, value in loss_components.items():
            metrics[loss_component] += value.item() * batch_size
        for metric_f in metric_fns:
            metrics[metric_f.__name__] += metric_f(output, target).item() * batch_size
            if compute_metrics_per_action:
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


def val(model, val_loader, loss_f, metric_fns, device, disable_tqdm=False, compute_metrics_per_action=True):
    model.eval()

    num_items = 0
    num_items_per_action = defaultdict(lambda: 0)

    metrics = defaultdict(lambda: 0)

    with torch.no_grad():
        for data in tqdm(val_loader, disable=disable_tqdm):
            data, embeddings, target = transform_batch(data)
            data = data.float().to(device)
            target = target.float().to(device)
            for k, v in embeddings.items():
                embeddings[k] = v.to(device)

            output = model(data, **embeddings)
            loss, loss_components = loss_f(output, target)

            batch_size = target.shape[0]
            metrics['loss'] += loss.item() * batch_size
            for loss_component, value in loss_components.items():
                metrics[loss_component] += value.item() * batch_size
            for metric_f in metric_fns:
                metrics[metric_f.__name__] += metric_f(output, target).item() * batch_size
                if compute_metrics_per_action:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        required=True,
        type=str,
        help='path to the configuration file'
    )
    parser.add_argument(
        '--num-dataset-items',
        required=False,
        type=int,
        default=None,
        help='number of items to form a dataset'
    )
    parser.add_argument(
        '--invert-rotations-train',
        action='store_true',
        help='indicates whether to invert rotation actions'
    )
    parser.add_argument(
        '--invert-rotations-val',
        action='store_true',
        help='indicates whether to invert rotation actions'
    )
    parser.add_argument(
        '--invert-collisions',
        action='store_true',
        help='indicates whether to invert rotation actions when the agent has collided with something'
    )
    parser.add_argument(
        '--not-use-turn-left',
        action='store_true',
    )
    parser.add_argument(
        '--not-use-turn-right',
        action='store_true',
    )
    parser.add_argument(
        '--not-use-move-forward',
        action='store_true',
    )
    parser.add_argument(
        '--not-use-rgb',
        action='store_true',
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_path = args.config_file

    config = get_config(config_path, new_keys_allowed=True)

    config.defrost()
    config.experiment_dir = os.path.join(config.log_dir, config.experiment_name)
    config.tb_dir = os.path.join(config.experiment_dir, 'tb')
    config.model.best_checkpoint_path = os.path.join(config.experiment_dir, 'best_checkpoint.pt')
    config.model.last_checkpoint_path = os.path.join(config.experiment_dir, 'last_checkpoint.pt')
    config.config_save_path = os.path.join(config.experiment_dir, 'config.yaml')

    config.train.dataset.params.num_points = args.num_dataset_items
    config.train.dataset.params.invert_rotations = args.invert_rotations_train
    config.train.dataset.params.invert_collisions = args.invert_collisions
    config.train.dataset.params.not_use_turn_left = args.not_use_turn_left
    config.train.dataset.params.not_use_turn_right = args.not_use_turn_right
    config.train.dataset.params.not_use_move_forward = args.not_use_move_forward
    config.train.dataset.params.not_use_rgb = args.not_use_rgb

    config.val.dataset.params.num_points = args.num_dataset_items
    config.val.dataset.params.invert_rotations = args.invert_rotations_val
    config.val.dataset.params.invert_collisions = args.invert_collisions
    config.val.dataset.params.not_use_turn_left = args.not_use_turn_left
    config.val.dataset.params.not_use_turn_right = args.not_use_turn_right
    config.val.dataset.params.not_use_move_forward = args.not_use_move_forward
    config.val.dataset.params.not_use_rgb = args.not_use_rgb
    config.freeze()

    # init distributed if run with torch.distributed.launch
    is_distributed = get_distrib_size()[2] > 1
    if is_distributed:
        local_rank, tcp_store = init_distrib_slurm(config.distrib_backend)
        if rank0_only():
            print("Initialized VO with {} workers".format(torch.distributed.get_world_size()))

        config.defrost()
        config.device = local_rank
        config.train.dataset.params.local_rank = local_rank
        config.train.dataset.params.world_size = torch.distributed.get_world_size()
        config.train.loader.is_distributed = False
        config.val.loader.is_distributed = True
        config.freeze()

    init_experiment(config)
    set_random_seed(config.seed)

    train_dataset = make_dataset(config.train.dataset)
    train_loader = make_data_loader(config.train.loader, train_dataset)
    train_metric_fns = make_metrics(config.train.metrics) if config.train.metrics else []

    val_dataset = make_dataset(config.val.dataset)
    val_loader = make_data_loader(config.val.loader, val_dataset)
    val_metric_fns = make_metrics(config.val.metrics) if config.val.metrics else []

    device = torch.device(config.device)
    model = make_model(config.model).to(device)
    if hasattr(config.model, 'pretrained_checkpoint') and config.model.pretrained_checkpoint is not None:
        model.load_state_dict(torch.load(config.model.pretrained_checkpoint, map_location=device))
    if is_distributed:
        model = init_distributed(model, device, find_unused_params=True)

    optimizer = make_optimizer(config.optim, model.parameters())
    scheduler = None

    loss_f = make_loss(config.loss)

    compute_metrics_per_action = config.compute_metrics_per_action
    batches_per_epoch = config.batches_per_epoch

    # TODO: fix tensorboard logging as in PPOTrainer ??
    train_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))

    model.train()

    num_items = 0
    num_items_per_action = defaultdict(lambda: 0)
    train_metrics = defaultdict(lambda: 0)

    for batch_index, data in enumerate(train_loader):
        data, embeddings, target = transform_batch(data)

        if hasattr(config, 'shuffle_train_batch') and config.shuffle_train_batch:
            indices = list(range(data.shape[0]))
            np.random.shuffle(indices)

            data = data[indices]
            target = target[indices]
            for k, v in embeddings.items():
                embeddings[k] = embeddings[k][indices]

        data = data.float().to(device)
        target = target.float().to(device)
        for k, v in embeddings.items():
            embeddings[k] = v.to(device)

        output = model(data, **embeddings)
        loss, loss_components = loss_f(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        train_metrics['loss'] += loss.item() * batch_size
        for loss_component, value in loss_components.items():
            train_metrics[loss_component] += value.item() * batch_size
        for metric_f in train_metric_fns:
            train_metrics[metric_f.__name__] += metric_f(output, target).item() * batch_size
            if compute_metrics_per_action:
                for action_id in embeddings['action'].unique():
                    action_name = action_id_to_action_name[action_id.item()]
                    action_mask = embeddings['action'] == action_id
                    action_metric_name = f'{metric_f.__name__}_{action_name}'
                    num_action_items = action_mask.sum()

                    action_metric_value = metric_f(output[action_mask], target[action_mask]).item()
                    train_metrics[action_metric_name] += action_metric_value * num_action_items
                    num_items_per_action[action_metric_name] += num_action_items

        num_items += batch_size

        if (batch_index + 1) % batches_per_epoch == 0:
            epoch = (batch_index + 1) // batches_per_epoch

            for metric_name in train_metrics:
                train_metrics[metric_name] /= num_items_per_action.get(metric_name, num_items)

            if is_distributed:
                train_metrics = coalesce_post_step(train_metrics, device)

            # report train metrics:
            write_metrics(epoch, train_metrics, train_writer)
            print_metrics('Train', train_metrics)

            # reset train metrics:
            num_items = 0
            num_items_per_action = defaultdict(lambda: 0)
            train_metrics = defaultdict(lambda: 0)

            # compute val metrics:
            val_metrics = val(model, val_loader, loss_f, val_metric_fns, device, is_distributed, compute_metrics_per_action)
            if is_distributed:
                val_metrics = coalesce_post_step(val_metrics, device)

            # report val metrics:
            write_metrics(epoch, val_metrics, val_writer)
            print_metrics('Val', val_metrics)

            if rank0_only() and config.model.save:
                best_checkpoint_path = config.model.best_checkpoint_path.replace('.pt', f'_{str(epoch).zfill(3)}e.pt')
                torch.save(model.state_dict(), best_checkpoint_path)
                print('Saved best model checkpoint to disk.')

            if scheduler:
                scheduler.step()

            model.train()

            if epoch > config.epochs:
                break

    train_writer.close()
    val_writer.close()

    if rank0_only() and config.model.save:
        torch.save(model.state_dict(), config.model.last_checkpoint_path)
        print('Saved last model checkpoint to disk.')


if __name__ == '__main__':
    main()