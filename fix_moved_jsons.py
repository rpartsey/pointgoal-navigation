import json
from glob import glob

dataset = 'gibson'
split = 'val'

source_dir = '/private/home/maksymets/pointgoal-navigation/data/vo_datasets'
target_dir = '/checkpoint/maksymets/data/vo_datasets'

for file_path in glob(f'{source_dir}/{dataset}/{split}/*.json'):
    with open(file_path, 'r') as file:
        content = json.load(file)

    for item in content['dataset']:
        item['source_frame_path'] = item['source_frame_path'].replace(source_dir, target_dir)
        item['target_frame_path'] = item['target_frame_path'].replace(source_dir, target_dir)
        item['source_depth_map_path'] = item['source_depth_map_path'].replace(source_dir, target_dir)
        item['target_depth_map_path'] = item['target_depth_map_path'].replace(source_dir, target_dir)

    with open(file_path, 'w') as file:
        json.dump(content, file)
