import os
import shutil
import argparse
from collections import defaultdict

import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter


from odometry.utils.early_stopping import EarlyStopping
from odometry.config.default import get_config

from odometry.models import make_model
from odometry.dataset import make_dataset, make_data_loader
from odometry.losses import make_loss
from odometry.optims import make_optimizer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    for k, v in metrics.items():
        writer.add_scalar(f'metrics/{k}', v, epoch)


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
    shutil.copy(config.self_path, config.config_save_path)


def train(model, optimizer, train_loader, loss_f, device):
    model.train()

    metrics = defaultdict(lambda: 0)

    for data in tqdm(train_loader):
        data, target = transform_batch(data)
        data = data.to(device).float()
        target = target.to(device).float()

        output = model(data)
        loss, loss_components = loss_f(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.shape[0]
        metrics['loss'] += loss.item() * batch_size
        for loss_component, value in loss_components.items():
            metrics[loss_component] += value.item() * batch_size

    dataset_length = len(train_loader.dataset)
    for metric_name in metrics:
        metrics[metric_name] /= dataset_length

    return metrics


def val(model, val_loader, loss_f, device):
    model.eval()

    metrics = defaultdict(lambda: 0)

    with torch.no_grad():
        for data in tqdm(val_loader):
            data, target = transform_batch(data)
            data = data.to(device).float()
            target = target.to(device).float()

            output = model(data)
            loss, loss_components = loss_f(output, target)

            batch_size = target.shape[0]
            metrics['loss'] += loss.item() * batch_size
            for loss_component, value in loss_components.items():
                metrics[loss_component] += value.item() * batch_size

    dataset_length = len(val_loader.dataset)
    for metric_name in metrics:
        metrics[metric_name] /= dataset_length

    return metrics


def transform_batch(batch):
    source_input, target_input = [], []

    source_images = batch['source_rgb']
    target_images = batch['target_rgb']
    source_input += [source_images]
    target_input += [target_images]

    source_depth_maps = batch['source_depth']
    target_depth_maps = batch['target_depth']
    source_input += [source_depth_maps]
    target_input += [target_depth_maps]

    concat_source_input = torch.cat(source_input, 1)
    concat_target_input = torch.cat(target_input, 1)
    transformed_batch = torch.cat(
        [
            concat_source_input,
            concat_target_input
        ],
        1
    )

    translation = batch['egomotion']['translation']
    rotation = batch['egomotion']['rotation'].view(translation.shape[0], -1)
    target = torch.cat(
        [
            translation,
            rotation
        ],
        1
    )

    return transformed_batch, target


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
        '--invert-rotations',
        action='store_true',
        help='indicates whether to invert rotation actions'
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
    config.self_path = config_path
    config.train.dataset.params.data_root = config.data_root
    config.train.dataset.params.num_points = args.num_dataset_items
    config.train.dataset.params.invert_rotations = args.invert_rotations
    config.val.dataset.params.data_root = config.data_root
    config.val.dataset.params.num_points = args.num_dataset_items
    config.val.dataset.params.invert_rotations = args.invert_rotations
    config.freeze()

    init_experiment(config)
    set_random_seed(config.seed)

    train_dataset = make_dataset(config.train.dataset)
    train_loader = make_data_loader(config.train.loader, train_dataset)

    val_dataset = make_dataset(config.val.dataset)
    val_loader = make_data_loader(config.val.loader, val_dataset)

    device = torch.device(config.device)
    model = make_model(config.model).to(device)

    optimizer = make_optimizer(config.optim, model.parameters())
    scheduler = None

    loss_f = make_loss(config.loss)

    early_stopping = EarlyStopping(
        **config.stopper.params
    )

    train_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'val'))

    for epoch in range(1, config.epochs + 1):
        print(f'Epoch {epoch}')
        train_metrics = train(model, optimizer, train_loader, loss_f, device)
        write_metrics(epoch, train_metrics, train_writer)
        print_metrics('Train', train_metrics)

        val_metrics = val(model, val_loader, loss_f, device)
        write_metrics(epoch, val_metrics, val_writer)
        print_metrics('Val', val_metrics)

        early_stopping(val_metrics['loss'])
        if config.model.save and early_stopping.counter == 0:
            torch.save(model.state_dict(), config.model.best_checkpoint_path)
            print('Saved best model checkpoint to disk.')
        if early_stopping.early_stop:
            print(f'Early stopping after {epoch} epochs.')
            break

        if scheduler:
            scheduler.step()

    train_writer.close()
    val_writer.close()

    if config.model.save:
        torch.save(model.state_dict(), config.model.last_checkpoint_path)
        print('Saved last model checkpoint to disk.')


if __name__ == '__main__':
    main()
