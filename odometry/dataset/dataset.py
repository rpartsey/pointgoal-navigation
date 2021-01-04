import json
from glob import glob
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from odometry.dataset.utils import get_relative_egomotion


class EgoMotionDataset(Dataset):
    def __init__(self, data_root, environment_dataset, split, transforms):
        super().__init__()
        self.data_root = data_root
        self.environment_dataset = environment_dataset
        self.split = split
        self.transforms = transforms
        self.meta_data = self._load_jsons()

    def _load_jsons(self):
        data = []

        for file_path in glob(f'{self.data_root}/{self.environment_dataset}/{self.split}/*.json'):
            with open(file_path, 'r') as file:
                scene_content = json.load(file)

            data += scene_content['dataset']

        return data

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

    @classmethod
    def from_config(cls, config, transforms):
        dataset_params = config.params
        return cls(
            data_root=dataset_params.data_root,
            environment_dataset=dataset_params.environment_dataset,
            split=dataset_params.split,
            transforms=transforms
        )


class EgoDataLoader(DataLoader):
    @classmethod
    def from_config(cls, config, dataset):
        loader_params = config.params
        return cls(
            dataset=dataset,
            batch_size=loader_params.batch_size,
            num_workers=loader_params.num_workers,
            shuffle=loader_params.shuffle,
        )
