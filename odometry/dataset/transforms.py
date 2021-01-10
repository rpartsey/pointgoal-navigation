import numpy as np

import torch
import torchvision.transforms as torch_transforms
from PIL import Image


class ConvertToTensor:
    def __call__(self, data):
        data = {
            k: (
                torch.from_numpy(np.asarray(v, dtype=np.float32))
                if 'rgb' in k
                else v
            )
            for k, v in data.items()
        }

        data = {
            k: (
                torch.from_numpy(np.asarray(v, dtype=np.float32))
                if 'depth' in k
                else v
            )
            for k, v in data.items()
        }

        data = {
            k: (
                {
                    'position': torch.from_numpy(np.asarray(v['position'], dtype=np.float32)),
                    'rotation': torch.from_numpy(np.asarray(v['rotation'], dtype=np.float32)),
                }
                if 'state' in k
                else v
            )
            for k, v in data.items()
        }

        data['egomotion']['translation'] = torch.from_numpy(
            np.asarray(
                data['egomotion']['translation'],
                dtype=np.float32
            )
        )

        return data


class PermuteChannels:
    def __call__(self, data):
        data = {
            k: (v.permute(2, 0, 1) if 'rgb' in k else v)
            for k, v in data.items()
        }
        data = {
            k: (v.permute(2, 0, 1) if 'depth' in k else v)
            for k, v in data.items()
        }
        return data


class Normalize:
    def __call__(self, data):
        data = {
            k: (v / 255. if 'rgb' in k else v)
            for k, v in data.items()
        }

        # normalizing ego-motion rotation (-10 and +10 degrees)
        rotation = data['egomotion']['rotation']
        if rotation > np.deg2rad(300):
            rotation -= (2 * np.pi)
        data['egomotion']['rotation'] = rotation

        return data


class Resize:
    def __init__(self, size, interpolation='BILINEAR'):
        self.resize = torch_transforms.Resize(size, getattr(Image, interpolation))

    def __call__(self, data):
        return {
            k: self.resize(v) if 'rgb' in k or 'depth' in k else v
            for k, v in data.items()
        }
