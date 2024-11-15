from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QComboBox, QGridLayout, QHBoxLayout, QPushButton, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image, ImageDraw
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from surround_view_dialog import SurroundViewDialog
from utils import pil2pixmap, get_ego_pose, map_to_screen
from utils import get_ego_pose  # utils.py에서 get_ego_pose 함수를 가져옵니다.
from nuscenes.utils.geometry_utils import view_points

class NuScenesViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # NuScenes 객체 생성
        self.nusc = NuScenes(version='v1.0-mini', dataroot='./v1.0-mini', verbose=True)
        
        # 현재 씬과 샘플 토큰 초기화
        self.scenes = self.nusc.scene
        self.current_scene_index = 0
        self.scene_token = self.scenes[self.current_scene_index]['token']
        self.scene = self.nusc.get('scene', self.scene_token)
        
        # 프레임에 대한 샘플 토큰 설정
        self.sample_tokens = [s['token'] for s in self.nusc.sample if s['scene_token'] == self.scene_token]
        self.current_frame = 0
        self.is_playing = False
        self.scale_factor = 1.0
        self.pan_offset = np.array([0, 0])
        self.loaded_images = {}
        self.surround_view_dialog = None

        # 카메라 센서 및 레이블 설정
        self.camera_sensors = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
        ]
        self.cam_labels = ["Front Left", "Front", "Front Right", "Back Left", "Back", "Back Right"]

        
        # UI 초기화 및 첫 프레임 강제 업데이트
        self.initUI()
        self.preload_surround_images()
        self.update_frame()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    # UI Initialization and Button Setup
    def initUI(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.scene_selector = QComboBox(self)
        for i, scene in enumerate(self.scenes):
            self.scene_selector.addItem(f"Scene {i + 1}: {scene['name']}")
        self.scene_selector.currentIndexChanged.connect(self.change_scene)
        main_layout.addWidget(self.scene_selector)

        viewer_layout = QGridLayout()
        self.frame_info_label = QLabel(f"Frame {self.current_frame + 1} / {len(self.sample_tokens)}", self)
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        viewer_layout.addWidget(self.frame_info_label, 0, 0, 1, 2)

        self.label_map = QLabel()
        self.label_map.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.label_map.setAlignment(Qt.AlignCenter)
        viewer_layout.addWidget(self.label_map, 1, 0, 1, 2)
        main_layout.addLayout(viewer_layout)

        button_layout = QHBoxLayout()
        self.add_buttons(button_layout)
        main_layout.addLayout(button_layout)
        self.setCentralWidget(main_widget)
        self.setWindowTitle("nuScenes Viewer with PyQt")
        self.show()

    # Adding Buttons
    def add_buttons(self, layout):
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play)
        layout.addWidget(self.btn_play)

        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause)
        layout.addWidget(self.btn_pause)

        self.btn_next = QPushButton("Next Frame")
        self.btn_next.clicked.connect(self.next_frame)
        layout.addWidget(self.btn_next)

        self.btn_prev = QPushButton("Previous Frame")
        self.btn_prev.clicked.connect(self.prev_frame)
        layout.addWidget(self.btn_prev)

        self.btn_show_surround = QPushButton("Show Surround View")
        self.btn_show_surround.clicked.connect(self.show_surround_view)
        layout.addWidget(self.btn_show_surround)

        self.btn_quit = QPushButton("Quit")
        self.btn_quit.clicked.connect(self.close)
        layout.addWidget(self.btn_quit)

    # 필요한 메서드 정의
    def play(self):
        """재생 시작 - 타이머 시작"""
        self.is_playing = True
        self.timer.start(100)  # 100ms 간격으로 업데이트

    def pause(self):
        """재생 일시 정지"""
        self.is_playing = False
        self.timer.stop()

    def next_frame(self):
        """다음 프레임으로 이동"""
        if self.current_frame < len(self.sample_tokens) - 1:
            self.current_frame += 1
            self.update_frame()

    def prev_frame(self):
        """이전 프레임으로 이동"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()
    def wheelEvent(self, event):
        """스크롤 시 확대/축소 비율 변경"""
        if event.angleDelta().y() > 0:
            self.scale_factor *= 1.1
        else:
            self.scale_factor /= 1.1
        self.update_frame()

    def mousePressEvent(self, event):
        """마우스 드래그 시작"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.last_mouse_position = event.pos()

    def mouseMoveEvent(self, event):
        """마우스 드래그로 pan_offset 업데이트"""
        if self.dragging:
            delta = event.pos() - self.last_mouse_position
            self.pan_offset += np.array([delta.x(), delta.y()])
            self.last_mouse_position = event.pos()
            self.update_frame()

    def mouseReleaseEvent(self, event):
        """마우스 드래그 종료"""
        if event.button() == Qt.LeftButton:
            self.dragging = False
            
    def show_surround_view(self):
        """Surround Viewer 열기"""
        sample_token = self.sample_tokens[self.current_frame]
        if sample_token in self.loaded_images:
            if self.surround_view_dialog is None or not self.surround_view_dialog.isVisible():
                # 새로운 Surround Viewer Dialog 생성
                self.surround_view_dialog = SurroundViewDialog(
                    self.loaded_images, sample_token, self.camera_sensors, self.cam_labels,
                    len(self.sample_tokens), self.current_frame, self
                )
                self.surround_view_dialog.setWindowTitle(
                    f"Scene {self.current_scene_index + 1}: {self.scenes[self.current_scene_index]['name']} - "
                    f"Frame {self.current_frame + 1} / {len(self.sample_tokens)}"
                )
                self.surround_view_dialog.setModal(False)
                self.surround_view_dialog.show()

    def preload_surround_images(self):
        for sample_token in self.sample_tokens:
            sample = self.nusc.get('sample', sample_token)
            self.loaded_images[sample_token] = {}
            for sensor in self.camera_sensors:
                if sensor in sample['data']:
                    data_token = sample['data'][sensor]
                    data_path = self.nusc.get_sample_data_path(data_token)
                    img = Image.open(data_path)
                    self.loaded_images[sample_token][sensor] = img
                    
    # Frame update, scene change, and drawing functions would be here
    def change_scene(self, index):
        """scenes 변경 기능"""
        # scenes 변경 전 Surround Viewer가 열려 있는지 확인하고 닫기
        if self.surround_view_dialog and self.surround_view_dialog.isVisible():
            self.surround_view_dialog.close()
            self.surround_view_dialog = None  # 참조 초기화

        # 새로운 scenes 설정
        self.current_scene_index = index
        self.scene_token = self.scenes[self.current_scene_index]['token']
        self.scene = self.nusc.get('scene', self.scene_token)
        self.sample_tokens = [s['token'] for s in self.nusc.sample if s['scene_token'] == self.scene_token]
        self.current_frame = 0
        self.loaded_images = {}  # 이전 scenes의 이미지를 지우고 초기화
        self.preload_surround_images()  # 새 scenes에 대해 surround view 이미지를 다시 로드
        self.update_frame()  # 새로운 scenes을 반영하여 Viewer 업데이트
        self.frame_info_label.setText(f"Frame {self.current_frame + 1} / {len(self.sample_tokens)}")

    def update_frame(self):
        """현재 프레임 업데이트"""
        if self.is_playing and self.current_frame < len(self.sample_tokens) - 1:
            self.current_frame += 1
        self.draw_map_and_objects()
        
        # 메인 Viewer 프레임 정보 업데이트
        self.frame_info_label.setText(f"Frame {self.current_frame + 1} / {len(self.sample_tokens)}")

        # Surround Viewer가 열려 있으면 업데이트
        if self.surround_view_dialog and self.surround_view_dialog.isVisible():
            sample_token = self.sample_tokens[self.current_frame]
            self.surround_view_dialog.update_surround_view(sample_token, self.current_frame)
            self.surround_view_dialog.setWindowTitle(
                f"Scene {self.current_scene_index + 1}: {self.scenes[self.current_scene_index]['name']} - "
                f"Frame {self.current_frame + 1} / {len(self.sample_tokens)}"
            )
            
    def draw_map_and_objects(self):
        try:
            width, height = 400, 400
            sample_token = self.sample_tokens[self.current_frame]
            # utils.py에서 가져온 get_ego_pose 함수를 사용하여 자차 위치와 방향 가져오기
            ego_position, ego_rotation = get_ego_pose(self.nusc, sample_token)
            
            img = Image.new("RGB", (width, height), "black")
            draw = ImageDraw.Draw(img)
            
            log = self.nusc.get('log', self.scene['log_token'])
            map_name = log['location']
            nusc_map = NuScenesMap(dataroot=self.nusc.dataroot, map_name=map_name)

            layers = ['drivable_area', 'road_segment']
            for layer_name in layers:
                records = getattr(nusc_map, layer_name)
                for record in records:
                    polygon_token = record.get('polygon_token')
                    if polygon_token:
                        polygon = nusc_map.extract_polygon(polygon_token)
                        polygon_center = np.mean(np.array(polygon.exterior.xy), axis=1)
                        distance = np.linalg.norm(polygon_center[:2] - ego_position)

                        if distance <= 50:
                            points = np.array(polygon.exterior.xy).T
                            for i in range(len(points) - 1):
                                x1, y1 = self.map_to_screen(points[i], width, height, ego_position)
                                x2, y2 = self.map_to_screen(points[i + 1], width, height, ego_position)
                                draw.line((x1, y1, x2, y2), fill=(200, 200, 200), width=2)

            # 객체 색상 설정
            object_colors = {
                'vehicle': (255, 0, 0),           # 빨간색
                'movable_object': (0, 0, 255),    # 파란색
                'human': (0, 255, 0),             # 초록색
            }

            sample = self.nusc.get('sample', sample_token)
            for ann_token in sample['anns']:
                box = self.nusc.get_box(ann_token)
                box_center = np.mean(box.corners(), axis=1)[:2]
                if np.linalg.norm(box_center - ego_position) <= 50:
                    corners = box.corners()[:3, :]
                    order = [0, 1, 5, 4, 0]
                    closed_corners = corners[:, order]
                    corners_2d = view_points(closed_corners, np.eye(3), normalize=False)[:2, :].T
                    
                    # 객체 종류 확인 및 색상 적용
                    object_type = box.name.split('.')[0]
                    color = object_colors.get(object_type, (255, 0, 0))  # 기본값 빨간색
                    
                    for i in range(len(corners_2d) - 1):
                        x1, y1 = self.map_to_screen(corners_2d[i], width, height, ego_position)
                        x2, y2 = self.map_to_screen(corners_2d[i + 1], width, height, ego_position)
                        draw.line((x1, y1, x2, y2), fill=color, width=2)

            ego_x, ego_y = self.map_to_screen(ego_position, width, height, ego_position)
            ego_rect = [(-5, -2.5), (-5, 2.5), (5, 2.5), (5, -2.5)]
            rotation_matrix = np.array([
                [ego_rotation.rotation_matrix[0, 0], ego_rotation.rotation_matrix[0, 1]],
                [-ego_rotation.rotation_matrix[1, 0], -ego_rotation.rotation_matrix[1, 1]]
            ])
            rotated_rect = [np.dot(rotation_matrix, point) * self.scale_factor + [ego_x, ego_y] for point in ego_rect]
            for i in range(4):
                x1, y1 = rotated_rect[i]
                x2, y2 = rotated_rect[(i + 1) % 4]
                draw.line((x1, y1, x2, y2), fill="white", width=2)

            # 이미지 변환 후 main viewer에 표시
            qt_img = self.pil2pixmap(img)
            self.label_map.setPixmap(qt_img)

        except Exception as e:
            print("Error in draw_map_and_objects:", e)

    def pil2pixmap(self, im):
        if im.mode == "RGB":
            r, g, b = im.split()
            im = Image.merge("RGB", (b, g, r))
        elif im.mode == "L":
            im = im.convert("RGB")
        im2 = im.convert("RGBA")
        data = im2.tobytes("raw", "RGBA")
        qimg = QImage(data, im.size[0], im.size[1], QImage.Format_ARGB32)
        return QPixmap.fromImage(qimg)

    def map_to_screen(self, point, width, height, ego_position):
        """화면 좌표 변환, 확대/축소 및 pan 적용"""
        scale = 2.0 * self.scale_factor
        x = int(-(point[0] - ego_position[0]) * scale + width / 2 + self.pan_offset[0])
        y = int((point[1] - ego_position[1]) * scale + height / 2 + self.pan_offset[1])
        return x, y
