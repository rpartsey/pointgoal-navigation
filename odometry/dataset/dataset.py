import copy
import json
import itertools
from glob import glob
from typing import Iterator
import multiprocessing as mp

import quaternion
import numpy as np
from PIL import Image

import torch
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader, IterableDataset

from habitat import get_config, make_dataset
from habitat.sims import make_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.tasks.nav.nav import merge_sim_episode_config

from odometry.dataset.utils import get_relative_egomotion


class EgoMotionDataset(Dataset):
    TURN_LEFT = 'TURN_LEFT'
    TURN_RIGHT = 'TURN_RIGHT'
    MOVE_FORWARD = 'MOVE_FORWARD'
    ROTATION_ACTIONS = ['TURN_LEFT', 'TURN_RIGHT']
    INVERSE_ACTION = {
        'TURN_LEFT': 'TURN_RIGHT',
        'TURN_RIGHT': 'TURN_LEFT'
    }
    ACTION_TO_ID = {
        'STOP': 0,
        'MOVE_FORWARD': 1,
        'TURN_LEFT': 2,
        'TURN_RIGHT': 3
    }

    def __init__(
            self,
            data_root,
            environment_dataset,
            split,
            transforms,
            num_points=None,
            invert_rotations=False,
            augmentations=None,
            not_use_turn_left=False,
            not_use_turn_right=False,
            not_use_move_forward=False,
            invert_collisions=False,
            not_use_rgb=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.environment_dataset = environment_dataset
        self.split = split
        self.transforms = transforms
        self.augmentations = augmentations
        self.not_use_turn_left = not_use_turn_left
        self.not_use_turn_right = not_use_turn_right
        self.not_use_move_forward = not_use_move_forward
        self.not_use_rgb = not_use_rgb
        self.jsons = self._load_jsons()
        self.invert_collisions = invert_collisions
        if invert_rotations:
            self._add_inverse_rotations()
        self.num_dataset_points = num_points or len(self.jsons)
        self.meta_data = self.jsons[:self.num_dataset_points]

    def _load_jsons(self):
        data = []

        for file_path in glob(f'{self.data_root}/{self.environment_dataset}/{self.split}/*.json'):
            with open(file_path, 'r') as file:
                scene_content = json.load(file)

            scene_dataset = scene_content['dataset']
            if self.not_use_turn_left:
                scene_dataset = [
                    frames for frames in scene_dataset
                    if frames['action'][0] != self.TURN_LEFT
                ]
            if self.not_use_turn_right:
                scene_dataset = [
                    frames for frames in scene_dataset
                    if frames['action'][0] != self.TURN_RIGHT
                ]
            if self.not_use_move_forward:
                scene_dataset = [
                    frames for frames in scene_dataset
                    if frames['action'][0] != self.MOVE_FORWARD
                ]

            data += scene_dataset

        return data

    def _add_inverse_rotations(self):
        new_jsons = []
        for item in self.jsons:
            new_jsons.append(item)
            action = item['action'][0]
            if action in self.ROTATION_ACTIONS:
                if item['collision'] and (not self.invert_collisions):
                    continue
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

        source_depth = np.load(meta['source_depth_map_path'])
        target_depth = np.load(meta['target_depth_map_path'])

        item = {
            'source_depth': source_depth,
            'target_depth': target_depth,
            'action': self.ACTION_TO_ID[meta['action'][0]] - 1,  # shift action ids by 1 as we don't use STOP
            'collision': int(meta['collision']),
            'egomotion': get_relative_egomotion(meta),
        }
        if not self.not_use_rgb:
            source_rgb = Image.open(meta['source_frame_path']).convert('RGB')
            target_rgb = Image.open(meta['target_frame_path']).convert('RGB')
            item['source_rgb'] = np.asarray(source_rgb)
            item['target_rgb'] = np.asarray(target_rgb)

        if self.augmentations is not None:
            item = self.augmentations(item)

        item = self.transforms(item)

        return item

    def __len__(self):
        return self.num_dataset_points

    @staticmethod
    def _swap_values(item, k1, k2):
        item[k1], item[k2] = item[k2], item[k1]

        return item

    @classmethod
    def from_config(cls, config, transforms, augmentations=None):
        dataset_params = config.params
        return cls(
            data_root=dataset_params.data_root,
            environment_dataset=dataset_params.environment_dataset,
            split=dataset_params.split,
            transforms=transforms,
            num_points=dataset_params.num_points,
            invert_rotations=dataset_params.invert_rotations,
            augmentations=augmentations,
            not_use_turn_left=dataset_params.not_use_turn_left,
            not_use_turn_right=dataset_params.not_use_turn_right,
            not_use_move_forward=dataset_params.not_use_move_forward,
            invert_collisions=dataset_params.invert_collisions,
            not_use_rgb=dataset_params.not_use_rgb
        )


class HSimDataset(IterableDataset):
    ACTION_TO_ID = {
        'STOP': 0,
        'MOVE_FORWARD': 1,
        'TURN_LEFT': 2,
        'TURN_RIGHT': 3
    }

    def __init__(
            self,
            config_file_path,
            steps_to_change_scene,
            transforms,
            augmentations=None,
            local_rank=None,
            world_size=None
    ):
        self.config_file_path = config_file_path
        self.steps_to_change_scene = steps_to_change_scene
        self.local_rank = local_rank
        self.world_size = world_size
        self.start = None
        self.stop = None
        self.sim = None
        self.config = None
        self.transforms = transforms
        self.augmentations = augmentations

    def __iter__(self) -> Iterator[T_co]:
        self.config = get_config(self.config_file_path)
        self.sim = make_sim(
            id_sim=self.config.SIMULATOR.TYPE,
            config=self.config.SIMULATOR
        )
        spf = ShortestPathFollower(
            sim=self.sim,
            goal_radius=self.config.TASK.SUCCESS.SUCCESS_DISTANCE,
            return_one_hot=False
        )
        dataset = make_dataset(
            id_dataset=self.config.DATASET.TYPE,
            config=self.config.DATASET
        )
        scene_ids = dataset.scene_ids
        # uniformly split scenes across torch DataLoader workers:
        self.split_scenes(num_scenes=len(scene_ids))

        scene_id_gen = itertools.cycle(scene_ids[self.start:self.stop])
        episode_gen = generate_pointnav_episode(
            sim=self.sim,
            is_gen_shortest_path=False
        )

        step = 0
        while True:
            if step % self.steps_to_change_scene == 0:
                current_scene_id = next(scene_id_gen)
                self.reconfigure_scene(current_scene_id)

                current_episode = next(episode_gen)
                self.reconfigure_episode(current_episode)
                self.sim.reset()

            action = spf.get_next_action(current_episode.goals[0].position)
            if action == self.ACTION_TO_ID['STOP']:
                current_episode = next(episode_gen)
                self.reconfigure_episode(current_episode)
                self.sim.reset()
                continue

            prev_observation = self.sim._prev_sim_obs
            prev_agent_state = self.sim.get_agent_state()

            observation = self.sim.step(action)
            agent_state = self.sim.get_agent_state()

            item = {
                'source_depth': np.expand_dims(prev_observation['depth'], 2),
                'target_depth': observation['depth'],
                'source_rgb': prev_observation['rgb'][:, :, :3],
                'target_rgb': observation['rgb'],
                'action': action - 1,  # shift action ids by 1 as we don't use STOP
                'collision': int(self.sim.previous_step_collided),
                'egomotion': get_relative_egomotion({
                    'source_agent_state': {
                        'position': prev_agent_state.position.tolist(),
                        'rotation': quaternion.as_float_array(prev_agent_state.rotation).tolist()
                    },
                    'target_agent_state': {
                        'position': agent_state.position.tolist(),
                        'rotation': quaternion.as_float_array(agent_state.rotation).tolist()
                    }
                })
            }
            if self.augmentations is not None:
                item = self.augmentations(item)

            item = self.transforms(item)

            step += 1
            yield item

    def reconfigure_scene(self, scene_id):
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene_id
        self.config.freeze()
        self.sim.reconfigure(self.config.SIMULATOR)

    def reconfigure_episode(self, episode):
        self.config.defrost()
        self.config.SIMULATOR = merge_sim_episode_config(
            self.config.SIMULATOR,
            episode
        )
        self.config.freeze()
        self.sim.reconfigure(self.config.SIMULATOR)

    @staticmethod
    def split_workload(start, stop, worker_id, num_workers):
        per_worker = int(np.ceil((stop - start) / num_workers))
        iter_start = worker_id * per_worker
        iter_stop = min(iter_start + per_worker, stop)

        return iter_start, iter_stop

    def split_scenes(self, num_scenes):
        if self.local_rank is None:
            distrib_worker_start, distrib_worker_stop = (0, num_scenes)
        else:
            distrib_worker_start, distrib_worker_stop = self.split_workload(
                start=0,
                stop=num_scenes,
                worker_id=self.local_rank,
                num_workers=self.world_size
            )

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.start, self.stop = (distrib_worker_start, distrib_worker_stop)
        else:
            self.start, self.stop = self.split_workload(
                start=distrib_worker_start,
                stop=distrib_worker_stop,
                worker_id=worker_info.id,
                num_workers=worker_info.num_workers
            )

    @classmethod
    def from_config(cls, config, transforms, augmentations=None):
        dataset_params = config.params
        return cls(
            config_file_path=dataset_params.config_file_path,
            steps_to_change_scene=dataset_params.steps_to_change_scene,
            transforms=transforms,
            augmentations=augmentations,
        )


class EgoDataLoader(DataLoader):
    @classmethod
    def from_config(cls, config, dataset, sampler):
        loader_params = config.params
        multiprocessing_context = loader_params.pop('multiprocessing_context', 'fork')
        return cls(
            dataset=dataset,
            sampler=sampler,
            multiprocessing_context=mp.get_context(multiprocessing_context),
            **loader_params
        )
