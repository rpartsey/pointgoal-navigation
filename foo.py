# np.linspace(1, 24, 6, endpoint=True)
#
# images_path = '/home/rpartsey/code/3d-navigation/pointgoal-navigation/navtrajectory/mp3d/val/colorized_topdown_map_v2'
#
# geodesic_distance_to_goal = []
#
# for file_path in glob(f'{images_path}/*.png'):
#     file_name = os.path.basename(file_path)
#     base_name = file_name[:-4]
#     distance_string = base_name.split('_')[-1]
#     _, distance_value = distance_string.split('=')
#     geodesic_distance_to_goal.append(float(distance_value))
#
# for file_path in glob(f'{images_path}/*.png'):
#     file_name = os.path.basename(file_path)
#     for value in np.unique(geodesic_distance_to_goal[(geodesic_distance_to_goal >= 22.75) & (geodesic_distance_to_goal < 30)]):
#         if f'geodesic_distance={str(value)}' in file_name:
#             print(file_name)
#
#
# for file_path in glob(f'{images_path}/*.png'):
#     file_name = os.path.basename(file_path)
#     for value in np.unique(geodesic_distance_to_goal[(geodesic_distance_to_goal<=8.25)]):
#         if f'geodesic_distance={str(value)}' in file_name:
#             print(file_name)

import os
from glob import glob
import shutil

base_video_dir = '/home/rpartsey/code/3d-navigation/pointgoal-navigation/navtrajectory/mp3d/val'
'video_dir'
'supplementary_video_dir'

gibson_episodes = [
    'Scioto_049',
    'Swormville_040',
    'Scioto_026',
    'Eastville_066',
    'Eastville_026',
    'Sisters_053',
    'Cantwell_038',
    'Cantwell_062',
    'Cantwell_052',
    'Mosquito_020',
    'Mosquito_036'
]

mp3d_episodes = [
    'X7HyMhZNoso_047',
    'x8F5xyUWy9e_045',
    'QUCTc6BB5sX_021',
    '2azQ1b91cZZ_013',
    'zsNo4HB9uLZ_002',
    '8194nk5LbLH_008',
    '2azQ1b91cZZ_008',
    'zsNo4HB9uLZ_006',
    'QUCTc6BB5sX_015',
    'Z6MFQCViBuw_006',
    'Z6MFQCViBuw_027',
    'Z6MFQCViBuw_045'
]

for file in glob(f'{base_video_dir}/video_dir/*.mp4'):
    for prefix in mp3d_episodes:
        if os.path.basename(file).startswith(prefix):
            shutil.copy(file, f'{base_video_dir}/supplementary_video_dir/{os.path.basename(file)}')