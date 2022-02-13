from collections import namedtuple

import numpy as np
import cv2
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField, RGBImageField

from odometry.dataset.dataset import EgoMotionDatasetResized
from odometry.dataset.utils import get_relative_egomotion


DatasetItem = namedtuple(
    'DatasetItem',
    [
        'source_depth',
        'source_rgb',
        'target_depth',
        'target_rgb',
        'action',
        'collision',
        'egomotion'
    ]
)


class FFCVEgoMotionDatasetResized(EgoMotionDatasetResized):
    """
        Wrapper class to return tuple instead of dict.

        The ffcv.writer.DatasetWriter expects dataset item to of type tuple.
    """
    def __getitem__(self, index):
        meta = self.metadata[index]

        egomotion = get_relative_egomotion(meta)
        rotation = egomotion['rotation']
        if rotation > np.deg2rad(300):
            rotation -= (2 * np.pi)
        elif rotation < -np.deg2rad(300):
            rotation += (2 * np.pi)
        egomotion['rotation'] = rotation

        item = DatasetItem(
            source_depth=self.read_depth(meta['source_depth_map_path']),
            source_rgb=self.read_rgb(meta['source_frame_path']),
            target_depth=self.read_depth(meta['target_depth_map_path']),
            target_rgb=self.read_rgb(meta['target_frame_path']),
            action=self.ACTION_TO_ID[meta['action'][0]] - 1,
            collision=int(meta['collision']),
            egomotion=np.append(egomotion['translation'], egomotion['rotation'].astype(np.float32))
        )

        return item

    @staticmethod
    def read_depth(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)


# RGB_SHAPE = (180, 320, 3)
# RGB_DTYPE = np.dtype(np.uint8)

DEPTH_SHAPE = (180, 320)
DEPTH_DTYPE = np.dtype(np.uint16)

EGOMOTION_SHAPE = (4,)
EGOMOTION_DTYPE = np.dtype(np.float32)

depth_field = NDArrayField(shape=DEPTH_SHAPE, dtype=DEPTH_DTYPE)
rgb_field = RGBImageField(write_mode='raw')

data_root = '/home/rpartsey/data/habitat/vo_datasets/hc_2021'
environment_dataset = 'gibson'
split = 'val'

dataset = FFCVEgoMotionDatasetResized(
    data_root,
    environment_dataset,
    split,
    # num_points=1000
)

write_path = '/home/rpartsey/data/habitat/vo_datasets/hc_2021/gibson_ffcv_format/val.beton'
num_workers = 32

writer = DatasetWriter(write_path, {
    'source_depth': NDArrayField(shape=DEPTH_SHAPE, dtype=DEPTH_DTYPE),
    'source_rgb': RGBImageField(write_mode='raw'),
    'target_depth': NDArrayField(shape=DEPTH_SHAPE, dtype=DEPTH_DTYPE),
    'target_rgb': RGBImageField(write_mode='raw'),
    'action': IntField(),
    'collision': IntField(),
    'egomotion': NDArrayField(shape=EGOMOTION_SHAPE, dtype=EGOMOTION_DTYPE)
}, num_workers=num_workers)

writer.from_indexed_dataset(dataset)
