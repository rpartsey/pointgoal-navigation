import os
import shutil
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir-path",
        required=True,
        type=str,
        help="Path to the directory with 'rgb', 'depth', 'json' folders.",
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    rgb_dir_path = os.path.join(args.data_dir_path, 'rgb')
    depth_dir_path = os.path.join(args.data_dir_path, 'depth')
    json_dir_path = os.path.join(args.data_dir_path, 'json')

    rgb_scene_dirs = os.listdir(rgb_dir_path)
    depth_scene_dirs = os.listdir(depth_dir_path)
    json_files = os.listdir(json_dir_path)

    started_scenes = list(set(rgb_scene_dirs).union(set(depth_scene_dirs)))
    finished_scenes = [file_name.split('.')[0] for file_name in json_files]
    not_finished_scenes = [scene for scene in started_scenes if scene not in finished_scenes]

    print(f'Started but not finished scenes: {not_finished_scenes}')
    for scene in not_finished_scenes:
        rgb_scene_dir_path = os.path.join(rgb_dir_path, scene)
        depth_scene_dir_path = os.path.join(depth_dir_path, scene)

        print(f'Deleting:\n\t* {rgb_scene_dir_path}\n\t* {depth_scene_dir_path}')
        shutil.rmtree(rgb_scene_dir_path)
        shutil.rmtree(depth_scene_dir_path)
