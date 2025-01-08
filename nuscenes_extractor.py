import traceback
from pathlib import Path
from typing import List

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix

from src.datamodule.nuscenes_data_utils import (
    OBJECT_TYPE_MAP,
    OBJECT_TYPE_MAP_COMBINED,
    LaneTypeMap,
    load_nuscenes_df,
)


class NuScenesExtractor:
    def __init__(
        self,
        radius: float = 150,
        save_path: Path = None,
        mode: str = "train",
        ignore_type: List[int] = [5, 6, 7, 8, 9],
        remove_outlier_actors: bool = True,
        dataset_path: str = "",
        version: str = "v1.0-mini",
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.remove_outlier_actors = remove_outlier_actors
        self.ignore_type = ignore_type
        self.nusc = NuScenes(version=version, dataroot=dataset_path, verbose=True)

    def save(self, scene_token: str):
        assert self.save_path is not None

        try:
            data = self.get_data(scene_token)
        except Exception:
            print(traceback.format_exc())
            print(f"Found error while extracting data from scene {scene_token}")
            return

        save_file = self.save_path / (scene_token + ".pt")
        torch.save(data, save_file)

    def get_data(self, scene_token: str):
        return self.process(scene_token)

    def process(self, scene_token: str):
        scene = self.nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']

        # Ego position and orientation initialization
        first_sample = self.nusc.get('sample', sample_token)
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', first_sample['data']['LIDAR_TOP'])['ego_pose_token'])
        origin = torch.tensor(ego_pose['translation'][:2], dtype=torch.float)
        theta = torch.tensor([ego_pose['rotation'][2]], dtype=torch.float)

        rotate_mat = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ])

        # Initialize tensors
        num_nodes = 50  # This value can be adjusted based on data
        x = torch.zeros(num_nodes, 110, 2, dtype=torch.float)
        x_attr = torch.zeros(num_nodes, 3, dtype=torch.uint8)
        x_heading = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_velocity = torch.zeros(num_nodes, 110, dtype=torch.float)
        x_track_horizon = torch.zeros(num_nodes, dtype=torch.int)
        padding_mask = torch.ones(num_nodes, 110, dtype=torch.bool)

        timestamps = []
        lane_positions, lane_ctrs, lane_angles, lane_attr, lane_padding_mask, is_intersections = [], [], [], [], [], []

        sample_token = scene['first_sample_token']
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])['ego_pose_token'])

            # Update timestamps
            timestamps.append(sample['timestamp'])

            # Process actors
            for ann_token in sample['anns']:
                annotation = self.nusc.get('sample_annotation', ann_token)
                position = torch.tensor(annotation['translation'][:2], dtype=torch.float)
                velocity = torch.tensor(annotation['velocity'][:2], dtype=torch.float)
                position = torch.matmul(position - origin, rotate_mat)

                # Example: Update actor tensors here

            # Process lanes (placeholder, requires NuScenes map API)
            # Example: Update lane-related tensors here

            sample_token = sample['next']

        # Process final tensors and return
        x_ctrs = x[:, 49, :2].clone()
        x_positions = x[:, :50, :2].clone()
        x_velocity_diff = x_velocity[:, :50].clone()

        x[:, 50:] = torch.where(
            (padding_mask[:, 49].unsqueeze(-1) | padding_mask[:, 50:]).unsqueeze(-1),
            torch.zeros(num_nodes, 60, 2),
            x[:, 50:] - x[:, 49].unsqueeze(-2),
        )
        x[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]).unsqueeze(-1),
            torch.zeros(num_nodes, 49, 2),
            x[:, 1:50] - x[:, :49],
        )
        x[:, 0] = torch.zeros(num_nodes, 2)

        x_velocity_diff[:, 1:50] = torch.where(
            (padding_mask[:, :49] | padding_mask[:, 1:50]),
            torch.zeros(num_nodes, 49),
            x_velocity_diff[:, 1:50] - x_velocity_diff[:, :49],
        )
        x_velocity_diff[:, 0] = torch.zeros(num_nodes)

        y = None if self.mode == "test" else x[:, 50:]

        return {
            "x": x[:, :50],
            "y": y,
            "x_attr": x_attr,
            "x_positions": x_positions,
            "x_centers": x_ctrs,
            "x_angles": x_heading,
            "x_velocity": x_velocity,
            "x_velocity_diff": x_velocity_diff,
            "x_padding_mask": padding_mask,
            "lane_positions": lane_positions,
            "lane_centers": lane_ctrs,
            "lane_angles": lane_angles,
            "lane_attr": lane_attr,
            "lane_padding_mask": lane_padding_mask,
            "is_intersections": is_intersections,
            "origin": origin.view(-1, 2),
            "theta": theta,
            "scenario_id": scene_token,
            "track_id": None,  # Placeholder for agent ID
            "city": "nuscenes",  # Placeholder for city equivalent
        }


# Usage Example
if __name__ == "__main__":
    extractor = NuScenesExtractor(
        dataset_path='/path/to/nuscenes',
        save_path=Path('./processed_data'),
    )

    scene_token = extractor.nusc.scene[0]['token']
    extractor.save(scene_token)
