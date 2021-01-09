from . import dataset as dataset_module
from . import transforms as transforms_module
from .transforms import build_transforms
from torchvision.transforms import Compose


def make_transforms(transforms_config):
    return Compose([
        getattr(transforms_module, transform_type)(**(config.params if config.params else {}))
        for transform_type, config in transforms_config.items()
    ])


def make_dataset(dataset_config):
    dataset_transforms = make_transforms(dataset_config.transforms)

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
