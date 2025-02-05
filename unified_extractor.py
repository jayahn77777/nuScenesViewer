import os
import torch
from pathlib import Path
from typing import List
from trajdata import UnifiedDataset
import numpy as np

class UnifiedExtractor:
    def __init__(
        self,
        save_path: Path = None,
        mode: str = "train",
        radius: float = 150,
    ) -> None:
        self.save_path = save_path
        self.mode = mode
        self.radius = radius
        self.dataset = UnifiedDataset(
            desired_data=["nusc_mini", "argo2_mini"],  # 사용할 데이터셋
            rebuild_cache=False,  # 캐시를 재사용
            rebuild_maps=False,  # 지도 데이터 재사용
            num_workers=os.cpu_count(),
            verbose=True,
        )

    def save(self, idx: int):
        assert self.save_path is not None

        try:
            data = self.process(idx)
            save_file = self.save_path / f"sample_{idx}.pt"
            torch.save(data, save_file)
        except Exception as e:
            print(f"Error processing index {idx}: {e}")

    def process(self, idx: int):
        sample = self.dataset[idx]  # AgentBatchElement 객체

        # 🔹 timestamps 속성 확인
        if hasattr(sample, "timestamps"):
            timestamps = sample.timestamps
        else:
            print(f"Warning: 'timestamps' not found in sample {idx}. Generating synthetic timestamps.")
            timestamps = np.linspace(sample.scene_ts - 5.0, sample.scene_ts + 5.0, 10)  # 10개의 프레임 생성

        # 🔹 ego 차량 정보 가져오기
        ego_pos = torch.tensor(sample.curr_agent_state_np.position[:2], dtype=torch.float)
        ego_heading = torch.tensor([sample.curr_agent_state_np.heading[0]], dtype=torch.float)

        rotate_mat = torch.tensor(
            [
                [torch.cos(ego_heading), -torch.sin(ego_heading)],
                [torch.sin(ego_heading), torch.cos(ego_heading)],
            ],
        )

        num_frames = len(timestamps)

        # 🔹 객체 정보 가져오기
        objects = getattr(sample, "neighbor_histories", [])
        num_objects = len(objects)

        x = torch.zeros(num_objects, num_frames, 2, dtype=torch.float)
        x_velocity = torch.zeros(num_objects, num_frames, dtype=torch.float)
        padding_mask = torch.ones(num_objects, num_frames, dtype=torch.bool)

        for i, obj in enumerate(objects):
            obj_pos = torch.tensor(obj[:, :2], dtype=torch.float)  # (N, 2)
            obj_heading = torch.zeros(len(obj))  # heading 정보가 없을 수도 있으므로 0으로 채움
            obj_velocity = torch.zeros(len(obj))  # velocity 정보도 없을 가능성 있음

            obj_pos_transformed = torch.matmul(obj_pos - ego_pos, rotate_mat)

            x[i, : len(obj_pos), :2] = obj_pos_transformed
            x_velocity[i, : len(obj_velocity)] = obj_velocity
            padding_mask[i, : len(obj_pos)] = False

        # 🔹 차선 정보 가져오기
        lane_positions = []
        lane_angles = []
        lane_padding_mask = []

        if hasattr(sample, "vec_map") and sample.vec_map is not None:
            for lane in sample.vec_map.lane_segments:
                lane_center = torch.tensor(lane.centerline[:, :2], dtype=torch.float)
                lane_center_transformed = torch.matmul(lane_center - ego_pos, rotate_mat)
                lane_positions.append(lane_center_transformed)

                angles = torch.atan2(
                    lane_center[1:, 1] - lane_center[:-1, 1],
                    lane_center[1:, 0] - lane_center[:-1, 0],
                )
                lane_angles.append(torch.cat([angles, angles[-1:]], dim=0))

        if lane_positions:
            lane_positions = torch.stack(lane_positions)
            lane_angles = torch.stack(lane_angles)
            lane_padding_mask = torch.zeros(lane_positions.shape[:2], dtype=torch.bool)
        else:
            lane_positions = torch.zeros(1, 20, 2)
            lane_angles = torch.zeros(1, 20)
            lane_padding_mask = torch.ones(1, 20, dtype=torch.bool)

        y = None if self.mode == "test" else x[:, 50:]

        return {
            "x": x[:, :50],
            "y": y,
            "x_positions": x[:, :50, :2],
            "x_angles": obj_heading[:50],
            "x_velocity": x_velocity,
            "x_padding_mask": padding_mask,
            "lane_positions": lane_positions,
            "lane_angles": lane_angles,
            "lane_padding_mask": lane_padding_mask,
            "origin": ego_pos.view(-1, 2),
            "theta": ego_heading,
            "timestamps": torch.tensor(timestamps, dtype=torch.float),
        }
