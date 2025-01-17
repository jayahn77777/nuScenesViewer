from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.utils.data_classes import Box

# NuScenes Object Type Mapping
OBJECT_TYPE_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 0,
    "vehicle.bus.rigid": 0,
    "vehicle.bus.bendy": 0,
    "vehicle.construction": 7,
    "vehicle.bicycle": 8,
    "vehicle.motorcycle": 2,
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "human.pedestrian.wheelchair": 1,
    "human.pedestrian.stroller": 1,
    "human.pedestrian.personal_mobility": 1,
    "human.pedestrian.police_officer": 1,
    "human.pedestrian.construction_worker": 1,
    "movable_object.barrier": 5,
    "movable_object.trafficcone": 5,
    "movable_object.pushable_pullable": 5,
    "static_object.bicycle_rack": 6,
}

# Simplified Object Type Mapping
OBJECT_TYPE_MAP_COMBINED = {
    "vehicle.car": 0,
    "vehicle.truck": 0,
    "vehicle.bus.rigid": 0,
    "vehicle.bus.bendy": 0,
    "vehicle.bicycle": 2,
    "vehicle.motorcycle": 2,
    "human.pedestrian.adult": 1,
    "human.pedestrian.child": 1,
    "movable_object.barrier": 3,
    "movable_object.trafficcone": 3,
    "static_object.bicycle_rack": 3,
    "unknown": 3,
}

# Lane Type Mapping (NuScenes-specific)
LaneTypeMap = {
    "vehicle_lane": 0,
    "bike_lane": 1,
    "bus_lane": 2,
}

def load_nusc_df(nusc: NuScenes, scene_token: str):
    """
    Load NuScenes scene data and map information.
    """
    scene = nusc.get('scene', scene_token)
    first_sample_token = scene['first_sample_token']
    map_name = scene['log_token']

    # Load map information
    map_path = f"{nusc.dataroot}/maps/{map_name}.json"
    nusc_map = NuScenesMap(dataroot=nusc.dataroot, map_name=map_name)

    # Collect all samples in the scene
    current_sample_token = first_sample_token
    samples = []

    while current_sample_token:
        sample = nusc.get('sample', current_sample_token)
        samples.append(sample)
        current_sample_token = sample['next']

    return samples, nusc_map, scene_token
