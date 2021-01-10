import copy
import json
from glob import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from odometry.dataset.utils import get_relative_egomotion


class EgoMotionDataset(Dataset):
    ROTATION_ACTIONS = ['TURN_LEFT', 'TURN_RIGHT']
    INVERSE_ACTION = {
        'TURN_LEFT': 'TURN_RIGHT',
        'TURN_RIGHT': 'TURN_LEFT'
    }

    def __init__(self, data_root, environment_dataset, split, transforms, num_points=None, invert_rotations=False):
        super().__init__()
        self.data_root = data_root
        self.environment_dataset = environment_dataset
        self.split = split
        self.transforms = transforms
        self.jsons = self._load_jsons()
        if invert_rotations:
            self._add_inverse_rotations()
        self.num_dataset_points = num_points or len(self.jsons)
        self.meta_data = self.jsons[:num_points]

    def _load_jsons(self):
        data = []

        for file_path in glob(f'{self.data_root}/{self.environment_dataset}/{self.split}/*.json'):
            with open(file_path, 'r') as file:
                scene_content = json.load(file)

            data += scene_content['dataset']

        return data

    def _add_inverse_rotations(self):
        new_jsons = []
        for item in self.jsons:
            new_jsons.append(item)
            action = item['action'][0]
            if action in self.ROTATION_ACTIONS:
                inv = copy.deepcopy(item)
                inv['action'][0] = self.INVERSE_ACTION[action]
                inv = self._swap_values(inv, 'source_frame_path', 'target_frame_path')
                inv = self._swap_values(inv, 'source_depth_map_path', 'target_depth_map_path')
                inv = self._swap_values(inv, 'source_agent_state', 'target_agent_state')
                new_jsons.append(inv)

        self.jsons = new_jsons

    def get_label(self, index):
        meta = self.meta_data[index]

        return meta['action'][0]

    def __getitem__(self, index):
        meta = self.meta_data[index]

        source_rgb = Image.open(meta['source_frame_path']).convert('RGB')
        target_rgb = Image.open(meta['target_frame_path']).convert('RGB')
        source_depth = np.load(meta['source_depth_map_path'])
        target_depth = np.load(meta['target_depth_map_path'])

        item = {
            'source_rgb': np.asarray(source_rgb),
            'target_rgb': np.asarray(target_rgb),
            'source_depth': source_depth,
            'target_depth': target_depth,
            'action': meta['action'][0],
            'egomotion': get_relative_egomotion(meta),
        }

        item = self.transforms(item)

        return item

    def __len__(self):
        return len(self.meta_data)

    @staticmethod
    def _swap_values(item, k1, k2):
        item[k1], item[k2] = item[k2], item[k1]

        return item

    @classmethod
    def from_config(cls, config, transforms):
        dataset_params = config.params
        return cls(
            data_root=dataset_params.data_root,
            environment_dataset=dataset_params.environment_dataset,
            split=dataset_params.split,
            transforms=transforms,
            num_points=dataset_params.num_points,
            invert_rotations=dataset_params.invert_rotations
        )


class EgoDataLoader(DataLoader):
    @classmethod
    def from_config(cls, config, dataset, sampler):
        loader_params = config.params
        return cls(
            dataset=dataset,
            batch_size=loader_params.batch_size,
            num_workers=loader_params.num_workers,
            shuffle=loader_params.shuffle,
            sampler=sampler
        )
