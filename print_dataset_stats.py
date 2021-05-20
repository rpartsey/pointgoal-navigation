import json
import numpy as np
from glob import glob
import argparse

import quaternion
from itertools import groupby

action_to_id_map = {
    'STOP': 0,
    'MOVE_FORWARD': 1,
    'TURN_LEFT': 2,
    'TURN_RIGHT': 3
}

id_to_action_map = {
    0: 'STOP',
    1: 'MOVE_FORWARD',
    2: 'TURN_LEFT',
    3: 'TURN_RIGHT'
}


def get_relative_egomotion(data, EPS=1e-8):
    '''
        Get agent states (source and target) from
        the simulator
    '''
    pos_s, rot_s = (
        np.asarray(
            data["source_agent_state"]["position"],
            dtype=np.float32
        ),
        np.asarray(
            data["source_agent_state"]["rotation"],
            dtype=np.float32
        )
    )

    pos_t, rot_t = (
        np.asarray(
            data["target_agent_state"]["position"],
            dtype=np.float32
        ),
        np.asarray(
            data["target_agent_state"]["rotation"],
            dtype=np.float32
        )
    )

    rot_s, rot_t = (
        quaternion.as_quat_array(rot_s),
        quaternion.as_quat_array(rot_t)
    )

    '''
        Convert source/target rotation arrays to 3x3 rotation matrices
    '''
    rot_s2w, rot_t2w = (
        quaternion.as_rotation_matrix(rot_s),
        quaternion.as_rotation_matrix(rot_t)
    )

    '''
        Construct the 4x4 transformation [agent-->world] matrices
        corresponding to the source and target agent state.
    '''
    trans_s2w, trans_t2w = (
        np.zeros(shape=(4, 4), dtype=np.float32),
        np.zeros(shape=(4, 4), dtype=np.float32)
    )
    trans_s2w[3, 3], trans_t2w[3, 3] = 1., 1.
    trans_s2w[0:3, 0:3], trans_t2w[0:3, 0:3] = rot_s2w, rot_t2w
    trans_s2w[0:3, 3], trans_t2w[0:3, 3] = pos_s, pos_t

    '''
        Construct the 4x4 transformation [world-->agent] matrices
        corresponding to the source and target agent state
        by inverting the earlier transformation matrices
    '''
    trans_w2s = np.linalg.inv(trans_s2w)
    trans_w2t = np.linalg.inv(trans_t2w)

    '''
        Construct the 4x4 transformation [target-->source] matrix
        (viewing things from the ref frame of source)
        -- take a point in the agent's coordinate at target state,
        -- transform that to world coordinates (trans_t2w)
        -- transform that to the agent's coordinates at source state (trans_w2s)
    '''
    trans_t2s = np.matmul(trans_w2s, trans_t2w)

    rotation = quaternion.as_rotation_vector(
        quaternion.from_rotation_matrix(trans_t2s[0:3, 0:3])
    )
    assert np.abs(rotation[0]) < EPS
    assert np.abs(rotation[2]) < EPS

    return {
        "translation": trans_t2s[0:3, 3],
        "rotation": rotation[1]
    }


def load_jsons(dataset_path):
    data = []
    for json_path in glob(f'{dataset_path}/*.json'):
        with open(json_path, 'r') as file:
            content = json.load(file)

        for item in content['dataset']:
            data.append({
                "source_frame_path": item["source_frame_path"],
                "target_frame_path": item["target_frame_path"],
                "source_depth_map_path": item["source_depth_map_path"],
                "target_depth_map_path": item["target_depth_map_path"],
                "label": {
                    "action": action_to_id_map[item['action'][0]],
                    "egomotion": get_relative_egomotion(item)
                },
                "source_agent_state": item["source_agent_state"],
                "target_agent_state": item["target_agent_state"],
                "info": {
                    "dataset": item["dataset"],
                    "scene": item["scene"],
                },
                'collision': int(item['collision'])
            })

    return data


def get_actions_distribution(scene_dataset):
    sorted_by_action = sorted(scene_dataset, key=lambda d: d['label']['action'])
    for action_id, group in groupby(sorted_by_action, key=lambda d: d['label']['action']):
        print(id_to_action_map[action_id], len(list(group)))


parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset-path',
    type=str,
    required=True
)
args = parser.parse_args()


vo_dataset_path = args.dataset_path
scene_dataset = load_jsons(vo_dataset_path)

print('Dataset length:', len(scene_dataset))
print('Actions distribution:')
get_actions_distribution(scene_dataset)
print('Number of collisions:', sum([item['collision'] for item in scene_dataset]))



