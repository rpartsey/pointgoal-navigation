import json
from glob import glob

dataset = 'gibson'
split = 'val'

old_path = '/private/home/maksymets/pointgoal-navigation/data/vo_datasets'  # path prefix taken from cat <any scene name>.json (this one should work)
target_dir = '/checkpoint/maksymets/data/vo_datasets'  # path to the directory where the dataset is physically located

for file_path in glob(f'{target_dir}/{dataset}/{split}/*.json'):
    with open(file_path, 'r') as file:
        content = json.load(file)

    for item in content['dataset']:
        item['source_frame_path'] = item['source_frame_path'].replace(old_path, target_dir)
        item['target_frame_path'] = item['target_frame_path'].replace(old_path, target_dir)
        item['source_depth_map_path'] = item['source_depth_map_path'].replace(old_path, target_dir)
        item['target_depth_map_path'] = item['target_depth_map_path'].replace(old_path, target_dir)

    with open(file_path, 'w') as file:
        json.dump(content, file)
