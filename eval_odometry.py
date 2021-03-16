import argparse
import os
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from habitat_extensions.config import get_config
from odometry.dataset import make_dataset, make_data_loader
from odometry.losses import make_loss
from odometry.metrics import make_metrics
from odometry.models import make_model
from odometry.utils import set_random_seed, transform_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--experiment-dir',
        required=True,
        type=str,
        help='path to the experiment logs dir'
    )
    parser.add_argument(
        '--checkpoint-name',
        required=True,
        type=str,
        nargs='+',
        help='one or more checkpoint names to evaluate the model'
    )
    parser.add_argument(
        '--logs-dir',
        required=True,
        type=str,
        help='path to the experiment logs dir'
    )
    parser.add_argument(
        '--batch-size',
        default=1,
        type=int,
        help='batch size'
    )
    parser.add_argument(
        '--num-workers',
        default=4,
        type=int,
        help='number of workers'
    )
    parser.add_argument(
        '--invert-rotations',
        action='store_true',
        help='indicates whether to invert rotation actions'
    )
    parser.add_argument(
        '--device-id',
        required=True,
        type=int,
        help='GPU id'
    )
    parser.add_argument(
        '--metrics',
        required=True,
        type=str,
        nargs='*',
        help='metrics names'
    )
    parser.add_argument(
        '--seed',
        default=2412,
        type=int,
        help='random seed'
    )
    args = parser.parse_args()

    return args


def print_metrics(phase, metrics):
    loss = metrics.pop('loss')

    loss_log_str = '{:6}loss: {:.6f}'.format(phase, loss)
    other_metrics_log_str = ' '.join([
        '{}: {:.6f}'.format(k, v)
        for k, v in metrics.items()
    ])

    metrics['loss'] = loss
    print(f'{loss_log_str} {other_metrics_log_str}')


def write_metrics(epoch, metrics, writer):
    for metric_name, value in metrics.items():
        key = 'losses' if 'loss' in metric_name else 'metrics'
        writer.add_scalar(f'{key}/{metric_name}', value, epoch)


def evaluate_checkpoint(args, config):
    seed = args.seed
    experiment_dir = args.experiment_dir
    checkpoint_names = args.checkpoint_name
    metric_names = args.metrics
    batch_size = args.batch_size
    logs_dir = args.logs_dir
    device = torch.device('cuda', args.device_id)

    set_random_seed(seed)

    dataset = make_dataset(config.val.dataset)
    loader = make_data_loader(config.val.loader, dataset)
    metric_fns = make_metrics(metric_names) if metric_names else []
    model = make_model(config.model).to(device)
    loss_f = make_loss(config.loss)

    for checkpoint_name in checkpoint_names:
        checkpoint_path = os.path.join(experiment_dir, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()

        epoch_metrics = defaultdict(lambda: 0)
        with SummaryWriter(log_dir=os.path.join(logs_dir, os.path.splitext(checkpoint_name)[0])) as writer:
            with torch.no_grad():
                for i, data in enumerate(tqdm(loader)):
                    batch_metrics = defaultdict(lambda: 0)

                    data, embeddings, target = transform_batch(data)
                    data = data.float().to(device)
                    target = target.float().to(device)
                    for k, v in embeddings.items():
                        embeddings[k] = v.to(device)

                    output = model(data, **embeddings)
                    loss, loss_components = loss_f(output, target)

                    batch_metrics['loss'] = loss.item()
                    for loss_component, value in loss_components.items():
                        batch_metrics[loss_component] = value.item()
                    for metric_f in metric_fns:
                        batch_metrics[metric_f.__name__] = metric_f(output, target).item()

                    write_metrics(i, batch_metrics, writer)

                    for k, v in batch_metrics.items():
                        epoch_metrics[k] += v * batch_size

        dataset_length = len(loader.dataset)
        for metric_name in epoch_metrics:
            epoch_metrics[metric_name] /= dataset_length

        print(checkpoint_name)
        print_metrics('Val', epoch_metrics)
        print()


def main():
    args = parse_args()
    config_path = os.path.join(args.experiment_dir, 'config.yaml')

    config = get_config(config_path)
    config.defrost()
    config.val.dataset.params.invert_rotations = args.invert_rotations
    config.val.loader.params.batch_size = args.batch_size
    config.val.loader.params.num_workers = args.num_workers
    config.freeze()

    evaluate_checkpoint(args, config)


if __name__ == '__main__':
    main()
