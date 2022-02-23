from torchvision.transforms import Compose
from torch.utils.data.distributed import DistributedSampler
from ffcv.loader import Loader as FFCVLoader, OrderOption
from .ffcv_transforms import *

from . import dataset as dataset_module
from . import transforms as transforms_module
from . import samplers as samplers_module
from . import augmentations as augmentations_module


def make_augmentations(augmentations_config):
    return Compose([
        getattr(augmentations_module, augmentation_type)(**(config.params if config.params else {}))
        for augmentation_type, config in augmentations_config.items()
    ])


def make_transforms(transforms_config):
    data_transforms = sorted(transforms_config.items(), key=lambda t: t[1].rank)

    return Compose([
        getattr(transforms_module, transform_type)(**(config.params if config.params else {}))
        for transform_type, config in data_transforms
    ])


def make_sampler(loader_config, dataset):
    sampler_name = loader_config.params.pop('sampler', None)
    if sampler_name:
        sampler_type = getattr(samplers_module, sampler_name)
        sampler = sampler_type(dataset)
    else:
        sampler = None

    return sampler


def make_dataset(dataset_config):
    dataset_transforms = make_transforms(dataset_config.transforms)
    dataset_augmentations = make_augmentations(dataset_config.augmentations) if dataset_config.augmentations else None

    dataset_type = getattr(dataset_module, dataset_config.type)
    dataset = dataset_type.from_config(
        dataset_config,
        dataset_transforms,
        dataset_augmentations
    )

    return dataset


def make_data_loader(config):
    if config.loader.type == 'FFCVLoader':
        pipelines = {
            'source_rgb': [SimpleRGBImageDecoder(), ToTensor(), TPermuteChannels(), TNormalizeRGB()],
            'source_depth': [NDArrayDecoder(), ToTensor(), TUnSqueeze(3), TPermuteChannels(), TNormalizeD()],
            'target_rgb': [SimpleRGBImageDecoder(), ToTensor(), TPermuteChannels(), TNormalizeRGB()],
            'target_depth': [NDArrayDecoder(), ToTensor(), TUnSqueeze(3), TPermuteChannels(), TNormalizeD()],
            'action': [IntDecoder(), ToTensor()],
            'collision': [IntDecoder(), ToTensor(),],
            'egomotion_translation': [NDArrayDecoder(), ToTensor()],
            'egomotion_rotation': [FloatDecoder(), ToTensor()]
        }
        loader = FFCVLoader(
            config.loader.params.fname,
            batch_size=config.loader.params.batch_size,
            num_workers=config.loader.params.num_workers,
            order=OrderOption[config.loader.params.order],
            pipelines=pipelines
        )
    else:
        dataset = make_dataset(config.dataset)
        if hasattr(config.loader, 'is_distributed') and config.loader.is_distributed:
            sampler = DistributedSampler(dataset)
            config.loader.params.pop('sampler', None)
        else:
            sampler = make_sampler(config.loader, dataset)

        data_loader_type = getattr(dataset_module, config.loader.type)
        loader = data_loader_type.from_config(
            config.loader,
            dataset,
            sampler
        )

    return loader
