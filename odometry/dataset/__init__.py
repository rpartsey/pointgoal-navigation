from torchvision.transforms import Compose

from . import dataset as dataset_module
from . import transforms as transforms_module
from . import samplers as samplers_module


def make_transforms(transforms_config):
    data_transforms = sorted(transforms_config.items(), key=lambda t: t[1].rank)

    return Compose([
        getattr(transforms_module, transform_type)(**(config.params if config.params else {}))
        for transform_type, config in data_transforms
    ])


def make_sampler(loader_config, dataset):
    sampler_name = loader_config.params.sampler
    if sampler_name:
        sampler_type = getattr(samplers_module, sampler_name)
        sampler = sampler_type(dataset)
    else:
        sampler = None

    return sampler


def make_dataset(dataset_config):
    dataset_transforms = make_transforms(dataset_config.transforms)

    dataset_type = getattr(dataset_module, dataset_config.type)
    dataset = dataset_type.from_config(
        dataset_config,
        dataset_transforms,
    )

    return dataset


def make_data_loader(loader_config, dataset):
    sampler = make_sampler(loader_config, dataset)

    data_loader_type = getattr(dataset_module, loader_config.type)
    loader = data_loader_type.from_config(
        loader_config,
        dataset,
        sampler
    )

    return loader