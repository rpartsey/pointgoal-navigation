import numpy as np
from habitat_sim.sensors.noise_models.redwood_depth_noise_model import RedwoodDepthNoiseModel
from habitat_sim.sensors.noise_models.gaussian_noise_model import GaussianNoiseModel


class RGBNoise:
    def __init__(self, intensity_constant):
        self.rgb_noise = GaussianNoiseModel(intensity_constant=intensity_constant)

    def __call__(self, item):
        item['source_rgb'] = self.rgb_noise.apply(item['source_rgb'])
        item['target_rgb'] = self.rgb_noise.apply(item['target_rgb'])
        return item


class DepthNoise:
    def __init__(self, noise_multiplier):
        self.depth_noise = RedwoodDepthNoiseModel(noise_multiplier=noise_multiplier)

    def __call__(self, item):
        item['source_depth'] = np.expand_dims(self.depth_noise.apply(np.squeeze(item['source_depth'], axis=2)), axis=2)
        item['target_depth'] = np.expand_dims(self.depth_noise.apply(np.squeeze(item['target_depth'], axis=2)), axis=2)
        return item
