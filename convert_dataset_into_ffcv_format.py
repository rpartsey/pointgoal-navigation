from collections import namedtuple

import numpy as np
import cv2
from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField, FloatField, RGBImageField

from odometry.dataset.dataset import EgoMotionDatasetResized


DatasetItemTuple = namedtuple(
    'DatasetItem',
    [
        'source_rgb',
        'source_depth',
        'target_rgb',
        'target_depth',
        'action',
        'collision',
        'egomotion_translation',
        'egomotion_rotation'
    ]
)


class FFCVEgoMotionDatasetResized(EgoMotionDatasetResized):
    """
        Wrapper class to return tuple instead of dict.

        The ffcv.writer.DatasetWriter expects dataset item to of type tuple.
    """
    def __getitem__(self, index):
        assert self.transforms is None, 'The iterable dataset should return raw, not transformed data'
        assert self.augmentations is None, 'The iterable dataset should return raw, not augmented data'

        item_dict = super(FFCVEgoMotionDatasetResized, self).__getitem__(index)

        if 'egomotion' in item_dict:
            rotation = item_dict['egomotion']['rotation']
            if rotation > np.deg2rad(300):
                rotation -= (2 * np.pi)
            elif rotation < -np.deg2rad(300):
                rotation += (2 * np.pi)
            item_dict['egomotion']['rotation'] = rotation

        item_tuple = DatasetItemTuple(
            source_rgb=item_dict['source_rgb'],
            source_depth=self.convert_depth(item_dict['source_depth']),
            target_rgb=item_dict['target_rgb'],
            target_depth=self.convert_depth(item_dict['target_depth']),
            action=item_dict['action'],
            collision=item_dict['collision'],
            egomotion_translation=np.ascontiguousarray(item_dict['egomotion']['translation']),
            egomotion_rotation=item_dict['egomotion']['rotation']
        )

        return item_tuple

    @staticmethod
    def read_depth(path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def convert_depth(depth):
        assert np.issubdtype(depth.dtype, np.uint16)
        dtype_min_val_diff = np.iinfo(np.uint16).min - np.iinfo(np.int16).min
        return (depth.astype(np.int32) - dtype_min_val_diff).astype(np.int16)


def main():
    data_root = '/home/rpartsey/data/habitat/vo_datasets/hc_2021'
    environment_dataset = 'gibson'
    split = 'val'

    dataset = FFCVEgoMotionDatasetResized(
        data_root,
        environment_dataset,
        split
    )

    DEPTH_SHAPE = (180, 320)
    DEPTH_DTYPE = np.dtype(np.int16)

    EGOMOTION_TRANSLATION_SHAPE = (3,)
    EGOMOTION_TRANSLATION_DTYPE = np.dtype(np.float32)

    write_path = '/home/rpartsey/data/habitat/vo_datasets/hc_2021/gibson_ffcv_format/val.beton'
    num_workers = -1

    writer = DatasetWriter(write_path, {
        'source_rgb': RGBImageField(write_mode='raw'),
        'source_depth': NDArrayField(shape=DEPTH_SHAPE, dtype=DEPTH_DTYPE),
        'target_rgb': RGBImageField(write_mode='raw'),
        'target_depth': NDArrayField(shape=DEPTH_SHAPE, dtype=DEPTH_DTYPE),
        'action': IntField(),
        'collision': IntField(),
        'egomotion_translation': NDArrayField(shape=EGOMOTION_TRANSLATION_SHAPE, dtype=EGOMOTION_TRANSLATION_DTYPE),
        'egomotion_rotation': FloatField()
    }, num_workers=num_workers)

    writer.from_indexed_dataset(dataset)


if __name__ == '__main__':
    main()
