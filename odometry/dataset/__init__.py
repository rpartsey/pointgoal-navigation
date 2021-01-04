from . import dataset as dataset_module
from .transforms import build_transforms


def make_dataset(dataset_config):
    dataset_transforms = build_transforms()

    dataset_type = getattr(dataset_module, dataset_config.type)
    dataset = dataset_type.from_config(
        dataset_config,
        dataset_transforms,
    )

    return dataset


def make_data_loader(loader_config, dataset):
    data_loader_type = getattr(dataset_module, loader_config.type)
    loader = data_loader_type.from_config(
        loader_config,
        dataset
    )

    return loader
