import os
import shutil
import argparse
from collections import defaultdict

from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from odometry.utils.early_stopping import EarlyStopping
from odometry.config.default import get_config
from odometry.models import make_model
from odometry.dataset import make_dataset, make_data_loader
from odometry.losses import make_loss
from odometry.optims import make_optimizer
from odometry.metrics import make_metrics
from odometry.utils import set_random_seed, transform_batch


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


def train(model, optimizer, train_loader, loss_f, metric_fns, device):
    model.train()

    metrics = defaultdict(lambda: 0)

    for data in tqdm(train_loader):
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

    dataset_length = len(train_loader.dataset)
    for metric_name in metrics:
        metrics[metric_name] /= dataset_length

    return metrics


def val(model, val_loader, loss_f, metric_fns, device):
    model.eval()

    metrics = defaultdict(lambda: 0)

    with torch.no_grad():
        for data in tqdm(val_loader):
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

    dataset_length = len(val_loader.dataset)
    for metric_name in metrics:
        metrics[metric_name] /= dataset_length

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
    config.val.dataset.params.num_points = args.num_dataset_items
    config.val.dataset.params.invert_rotations = args.invert_rotations_val
    config.freeze()

    init_experiment(config)
    set_random_seed(config.seed)

    train_dataset = make_dataset(config.train.dataset)
    train_loader = make_data_loader(config.train.loader, train_dataset)
    train_metric_fns = make_metrics(config.train.metrics) if config.train.metrics else []

    if hasattr(config, 'train_val'):
        train_val_dataset = make_dataset(config.train_val.dataset)
        train_val_loader = make_data_loader(config.train_val.loader, train_val_dataset)
        train_val_metric_fns = make_metrics(config.train_val.metrics) if config.train_val.metrics else []

    val_dataset = make_dataset(config.val.dataset)
    val_loader = make_data_loader(config.val.loader, val_dataset)
    val_metric_fns = make_metrics(config.val.metrics) if config.val.metrics else []

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
    if hasattr(config, 'train_val'):
        train_val_writer = SummaryWriter(log_dir=os.path.join(config.tb_dir, 'train_val'))

    for epoch in range(1, config.epochs + 1):
        print(f'Epoch {epoch}')
        train_metrics = train(model, optimizer, train_loader, loss_f, train_metric_fns, device)
        write_metrics(epoch, train_metrics, train_writer)
        print_metrics('Train', train_metrics)

        val_metrics = val(model, val_loader, loss_f, val_metric_fns, device)
        write_metrics(epoch, val_metrics, val_writer)
        print_metrics('Val', val_metrics)

        if hasattr(config, 'train_val'):
            train_val_metrics = val(model, train_val_loader, loss_f, train_val_metric_fns, device)
            write_metrics(epoch, train_val_metrics, train_val_writer)
            print_metrics('Train-val', train_val_metrics)

        early_stopping(val_metrics['loss'])
        if config.model.save and early_stopping.counter == 0:
            best_checkpoint_path = config.model.best_checkpoint_path.replace('.pt', f'_{str(epoch).zfill(2)}e.pt')
            torch.save(model.state_dict(), best_checkpoint_path)
            print('Saved best model checkpoint to disk.')
        if early_stopping.early_stop:
            print(f'Early stopping after {epoch} epochs.')
            break

        if scheduler:
            scheduler.step()

    train_writer.close()
    val_writer.close()
    if hasattr(config, 'train_val'):
        train_val_writer.close()

    if config.model.save:
        torch.save(model.state_dict(), config.model.last_checkpoint_path)
        print('Saved last model checkpoint to disk.')


if __name__ == '__main__':
    main()
