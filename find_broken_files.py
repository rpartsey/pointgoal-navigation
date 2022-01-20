import argparse
import gzip
import json
import os.path
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-root-dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '--split',
        type=str,
        required=True
    )
    parser.add_argument(
        '--fix-broken',
        action='store_true'
    )
    args = parser.parse_args()

    dataset_name = 'gibson'
    dataset_root_dir = args.dataset_root_dir
    split = args.split
    fix_broken = args.fix_broken

    dataset_path = f'{dataset_root_dir}/{dataset_name}/{split}'
    json_dir = f'{dataset_path}/json'

    print('Scenes that contain broken observation pairs:')
    for json_path in sorted(glob(f'{json_dir}/*.json.gz')):
        with gzip.open(json_path, 'rt') as file:
            content = json.loads(file.read())

        scene_name = os.path.basename(json_path)[:-len('.json.gz')]
        n_scene_obs = len(content['dataset'])
        n_invalid_obs = 0

        valid_items = []
        for item in content['dataset']:
            obs_is_valid = bool(
                os.path.exists(item['source_frame_path']) and
                os.path.exists(item['target_frame_path']) and
                os.path.exists(item['source_depth_map_path']) and
                os.path.exists(item['target_depth_map_path'])
            )

            if obs_is_valid:
                valid_items.append(item)

            n_invalid_obs += int(not obs_is_valid)

        if n_invalid_obs > 0:
            print(f'{scene_name:<20} total obs pairs: {n_scene_obs:<10} invalid pairs {(n_invalid_obs/n_scene_obs) * 100:0.3f} %')

            if fix_broken:
                with gzip.open(json_path, 'wt') as f:
                    content['dataset'] = valid_items
                    json.dump(content, f)

                print(f'Fixed: {scene_name}!')


main()
