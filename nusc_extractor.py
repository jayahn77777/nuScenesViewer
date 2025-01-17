import os
import pickle
import torch
from pathlib import Path
from nuscenes.nuscenes import NuScenes

class NuscExtractor:
    def __init__(self, input_path, save_path, mode):
        self.input_path = input_path
        self.save_path = Path(save_path)
        self.mode = mode
        self.nusc = NuScenes(version='v1.0-mini', dataroot=input_path, verbose=True)

    def process(self):
        os.makedirs(self.save_path, exist_ok=True)

        for scene in self.nusc.scene:
            scene_name = scene['name']
            scenario_id = scene['token']  # 장면 토큰을 시나리오 ID로 사용
            #print(f"Processing scenario: {scene_name}")

            first_sample_token = scene['first_sample_token']
            current_sample_token = first_sample_token

            # 개별 샘플 데이터와 메타데이터 저장
            scene_data = {}
            bin_data = {}

            index = 0

            while current_sample_token:
                sample = self.nusc.get('sample', current_sample_token)
                data_entry = self.process_sample(sample)

                # 개별 샘플 데이터를 메모리에 저장
                bin_data[f"{index}.bin"] = pickle.dumps(data_entry)
                scene_data[f"{index}.bin"] = data_entry

                current_sample_token = sample['next']
                index += 1

            # 전체 데이터를 .pt 파일에 저장
            scenario_path = self.save_path / f"scenario_{scenario_id}.pt"
            torch.save({
                'byteorder': 'little',
                'version': 'v1.0-mini',
                'data': bin_data,       # .bin 데이터 포함
                'metadata': scene_data  # 메타데이터 포함
            }, scenario_path)


    # def process(self):
    #     os.makedirs(self.save_path, exist_ok=True)

    #     for scene in self.nusc.scene:
    #         scene_name = scene['name']
    #         scenario_id = scene['token']  # 장면 토큰을 시나리오 ID로 사용
    #         print(f"Processing scenario: {scene_name}")

    #         first_sample_token = scene['first_sample_token']
    #         current_sample_token = first_sample_token

    #         # 저장 경로 생성
    #         scenario_path = self.save_path / f"scenario_{scenario_id}.pt"
    #         bin_data_path = self.save_path / f"scenario_{scenario_id}_data"
    #         os.makedirs(bin_data_path, exist_ok=True)

    #         scene_data = []
    #         index = 0

    #         while current_sample_token:
    #             sample = self.nusc.get('sample', current_sample_token)
    #             data_entry = self.process_sample(sample)

    #             # Save individual sample data as .bin
    #             bin_file_path = bin_data_path / f"{index}.bin"
    #             with open(bin_file_path, 'wb') as bin_file:
    #                 pickle.dump(data_entry, bin_file)
    #             scene_data.append(str(bin_file_path))

    #             current_sample_token = sample['next']
    #             index += 1

    #         # Save metadata as .pkl
    #         pkl_file_path = bin_data_path / 'data.pkl'
    #         with open(pkl_file_path, 'wb') as pkl_file:
    #             pickle.dump(scene_data, pkl_file)

    #         # Save additional metadata as .pt
    #         torch.save({
    #             'byteorder': 'little',
    #             'version': 'v1.0-mini',
    #             'data_folder': str(bin_data_path),
    #             'data_list': scene_data
    #         }, scenario_path)

    def process_sample(self, sample):
        """
        Process a single NuScenes sample and extract relevant data.
        """
        # 샘플의 기본 정보 추출
        sample_data = {
            'timestamp': sample['timestamp'],
            'token': sample['token'],
            'ego_pose': self.extract_ego_pose(sample),
            'annotations': [
                self.extract_annotation(annotation_token)
                for annotation_token in sample['anns']
            ],
        }
        return sample_data

    def extract_annotation(self, annotation_token):
        """
        Extract detailed information about an annotation.
        """
        annotation = self.nusc.get('sample_annotation', annotation_token)

        return {
            'instance_token': annotation['instance_token'],
            'category_name': annotation['category_name'],
            'translation': annotation['translation'],
            'rotation': annotation['rotation'],
            'size': annotation['size'],
            'num_lidar_pts': annotation['num_lidar_pts'],
            'num_radar_pts': annotation['num_radar_pts'],
    }
    def extract_ego_pose(self, sample):
        lidar_data_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_data_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

        return {
            'translation': ego_pose['translation'],
            'rotation': ego_pose['rotation']
        }