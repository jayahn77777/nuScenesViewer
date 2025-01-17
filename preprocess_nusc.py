import os
import json
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

# Configuration
NUSCENES_ROOT = "./data_nusce_v1.0-mini"  # Update this path
OUTPUT_DIR = "./outputs_nusc"  # Update this path
VERSION = "v1.0-mini"  # Adjust as necessary

# Initialize NuScenes
def initialize_nuscenes(version, root):
    return NuScenes(version=version, dataroot=root, verbose=True)

# Process Scene Data
def process_scene(nusc, scene_token, output_dir):
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']

    all_samples = []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        
        sample_data = {
            'timestamp': sample['timestamp'],
            'ego_pose': get_ego_pose(nusc, sample),
            'objects': get_objects(nusc, sample)
        }
        all_samples.append(sample_data)

        sample_token = sample['next'] if sample['next'] else None

    output_path = os.path.join(output_dir, f"scene_{scene_token}.json")
    with open(output_path, 'w') as f:
        json.dump(all_samples, f, indent=4)

# Get Ego Pose
def get_ego_pose(nusc, sample):
    ego_pose_token = nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    
    return {
        'position': ego_pose['translation'],
        'rotation': ego_pose['rotation']
    }

# Get Objects
def get_objects(nusc, sample):
    objects = []
    for ann_token in sample['anns']:
        annotation = nusc.get('sample_annotation', ann_token)
        objects.append({
            'category': annotation['category_name'],
            'position': annotation['translation'],
            'size': annotation['size'],
            'rotation': annotation['rotation'],
            'velocity': get_velocity(nusc, annotation)
        })
    return objects

# Get Velocity
def get_velocity(nusc, annotation):
    velocity = annotation['velocity']  # velocity is precomputed in nuScenes annotations
    return velocity if velocity else [0.0, 0.0]

# Main Function
def main():
    nusc = initialize_nuscenes(VERSION, NUSCENES_ROOT)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for scene in nusc.scene:
        print(f"Processing scene: {scene['name']}")
        process_scene(nusc, scene['token'], OUTPUT_DIR)

if __name__ == "__main__":
    main()
