import os
import json
import numpy as np 
from glob import glob

# 1. CHANGE /home/rpartsey/data/habitat/vo_datasets/noisy to your location
vo_dataset_path1 = '/home/rpartsey/data/habitat/vo_datasets/noisy/gibson/{split}'

# 2. CHANGE /home/rpartsey/data/habitat/vo_datasets/noisy_big to your location
vo_dataset_path2 = '/home/rpartsey/data/habitat/vo_datasets/noisy_big/gibson/{split}'

vo_train_dataset_path1 = vo_dataset_path1.format(split='train')
vo_train_dataset_path2 = vo_dataset_path2.format(split='train')

assert os.path.exists(vo_train_dataset_path1)
assert os.path.exists(vo_train_dataset_path2)

# 3. CHANGE the part of path that is different between vo_dataset_path1 and vo_dataset_path2 to the new directory where merged files will be stored
# in my case its 'noisy' 
# in your case its 'vo_datasets' if i'm not mistaken
os.makedirs(vo_train_dataset_path1.replace('noisy', 'noisy_plus_noisy_big'))

for json_meta_file_path1 in sorted(glob(f'{vo_train_dataset_path1}/*.json')):
    # 4. USE 'vo_datasets', 'vo_datasets2' instead of 'noisy', 'noisy_big'
    json_meta_file_path2 = json_meta_file_path1.replace('noisy', 'noisy_big')
    
    print(os.path.basename(json_meta_file_path1))
    
    merged_content = {'dataset': []}
    for json_meta_file_path in [json_meta_file_path1, json_meta_file_path2]:
        with open(json_meta_file_path, 'r') as json_meta_file:
            content = json.load(json_meta_file)
        merged_content['dataset'] += content['dataset']
        
    # 5. CHANGE The same as in step 3
    with open(json_meta_file_path1.replace('noisy', 'noisy_plus_noisy_big'), 'w') as dest_file:
        json.dump(merged_content, dest_file)
