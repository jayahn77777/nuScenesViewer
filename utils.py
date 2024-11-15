from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

def pil2pixmap(im):
    if im.mode == "RGB":
        r, g, b = im.split()
        im = Image.merge("RGB", (b, g, r))
    elif im.mode == "L":
        im = im.convert("RGB")
    im2 = im.convert("RGBA")
    data = im2.tobytes("raw", "RGBA")
    qimg = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
    return QPixmap.fromImage(qimg)

def map_to_screen(point, width, height, ego_position, scale_factor, pan_offset):
    scale = 2.0 * scale_factor
    x = int(-(point[0] - ego_position[0]) * scale + width / 2 + pan_offset[0])
    y = int((point[1] - ego_position[1]) * scale + height / 2 + pan_offset[1])
    return x, y

def get_ego_pose(nusc, sample_token):
    sample = nusc.get('sample', sample_token)
    sample_data_token = sample['data']['CAM_FRONT']
    sample_data = nusc.get('sample_data', sample_data_token)
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    position = ego_pose['translation'][:2]
    rotation = Quaternion(ego_pose['rotation'])
    return position, rotation
