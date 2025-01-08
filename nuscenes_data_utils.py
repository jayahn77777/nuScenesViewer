from pathlib import Path
import pandas as pd
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

OBJECT_TYPE_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 0,
    "vehicle.bus": 0,
    "vehicle.bicycle": 2,
    "vehicle.motorcycle": 2,
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "human.pedestrian.wheelchair": 1,
    "human.pedestrian.stroller": 1,
    "human.pedestrian.personal_mobility": 1,
    "static.object.bicycle_rack": 3,
    "movable_object.barrier": 3,
    "movable_object.trafficcone": 3,
    "movable_object.debris": 3,
    "movable_object.pushable_pullable": 3,
    "vehicle.emergency.police": 0,
    "vehicle.emergency.ambulance": 0,
}

OBJECT_TYPE_MAP_COMBINED = {
    "vehicle.car": 0,
    "vehicle.truck": 0,
    "vehicle.bus": 0,
    "vehicle.bicycle": 2,
    "vehicle.motorcycle": 2,
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "human.pedestrian.wheelchair": 1,
    "human.pedestrian.stroller": 1,
    "human.pedestrian.personal_mobility": 1,
    "static.object.bicycle_rack": 3,
    "movable_object.barrier": 3,
    "movable_object.trafficcone": 3,
    "movable_object.debris": 3,
    "movable_object.pushable_pullable": 3,
    "vehicle.emergency.police": 0,
    "vehicle.emergency.ambulance": 0,
}

LaneTypeMap = {
    "road_segment": 0,
    "pedestrian_crossing": 1,
    "intersection": 2,
}

def load_nuscenes_df(dataset_path: str, version: str = "v1.0-mini"):
    nusc = NuScenes(version=version, dataroot=dataset_path, verbose=True)

    scenes = []
    for scene in nusc.scene:
        scene_token = scene['token']
        first_sample_token = scene['first_sample_token']
        samples = []

        sample_token = first_sample_token
        while sample_token:
            sample = nusc.get('sample', sample_token)
            samples.append(sample)
            sample_token = sample['next']

        scenes.append({
            "scene_token": scene_token,
            "samples": samples
        })

    return scenes
