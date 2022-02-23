import numpy as np
import torch
from torch import Tensor
from ffcv.transforms import ToTensor
from ffcv.fields.decoders import NDArrayDecoder, FloatDecoder, IntDecoder, SimpleRGBImageDecoder

from convert_dataset_into_ffcv_format import DatasetItemTuple


def transform_batch_ffcv(batch):
    batch = DatasetItemTuple(*batch)
    source_input, target_input = [], []

    source_depth_maps = batch.source_depth
    target_depth_maps = batch.target_depth
    source_input += [source_depth_maps]
    target_input += [target_depth_maps]

    if 'source_rgb' in batch._fields:
        source_images = batch.source_rgb
        target_images = batch.target_rgb
        source_input += [source_images]
        target_input += [target_images]

    if all(key in batch._fields for key in ['source_depth_discretized', 'target_depth_discretized']):
        source_d_depth = batch.source_depth_discretized
        target_d_depth = batch.target_depth_discretized
        source_input += [source_d_depth]
        target_input += [target_d_depth]

    concat_source_input = torch.cat(source_input, 1)
    concat_target_input = torch.cat(target_input, 1)
    transformed_batch = torch.cat(
        [
            concat_source_input,
            concat_target_input
        ],
        1
    )

    if all(key in batch._fields for key in ['egomotion_translation', 'egomotion_rotation']):
        target = torch.cat(
            [
                batch.egomotion_translation,
                batch.egomotion_rotation.view(batch.egomotion_translation.shape[0], -1)
            ],
            1
        )
    else:
        target = None

    embeddings = {}
    if 'action' in batch._fields:
        embeddings['action'] = batch.action.squeeze(1)

    if 'collision' in batch._fields:
        embeddings['collision'] = batch.collision.squeeze(1)

    return transformed_batch, embeddings, target


class TPermuteChannels(torch.nn.Module):
    def __init__(self, permutation=(0, 3, 1, 2)):
        super().__init__()
        self._permutation = permutation

    def forward(self, tensor: Tensor):
        return tensor.permute(*self._permutation)


class TNormalizeRGB(torch.nn.Module):
    def __init__(self, scale=np.iinfo(np.uint8).max):
        super().__init__()
        self._scale = scale

    def forward(self, tensor: Tensor):
        return tensor / self._scale


class TNormalizeD(torch.nn.Module):
    def __init__(self, scale=np.iinfo(np.uint16).max):
        super().__init__()
        self._scale = scale
        self._dtype_min_val_diff = np.iinfo(np.uint16).min - np.iinfo(np.int16).min

    def forward(self, tensor: Tensor):
        return (tensor + self._dtype_min_val_diff) / self._scale


class TUnSqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    def forward(self, tensor: Tensor):
        return tensor.unsqueeze(self._dim)
