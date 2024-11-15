from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
from utils import pil2pixmap

class SurroundViewDialog(QDialog):
    def __init__(self, loaded_images, sample_token, camera_sensors, cam_labels, total_frames, current_frame, parent=None):
        super().__init__(parent)
        self.loaded_images = loaded_images
        self.sample_token = sample_token
        self.camera_sensors = camera_sensors
        self.cam_labels = cam_labels
        self.total_frames = total_frames
        self.current_frame = current_frame
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()
        self.frame_info_label = QLabel(f"Frame {self.current_frame + 1} / {self.total_frames}", self)
        self.frame_info_label.setAlignment(Qt.AlignCenter)
        self.frame_info_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        main_layout.addWidget(self.frame_info_label)

        layout = QGridLayout()
        self.labels = {}
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for i, sensor in enumerate(self.camera_sensors):
            label = QLabel()
            label.setAlignment(Qt.AlignCenter)
            cam_text = QLabel(self.cam_labels[i], self)
            cam_text.setAlignment(Qt.AlignCenter)
            cam_text.setStyleSheet("color: black; font-weight: bold;")
            layout.addWidget(cam_text, positions[i][0] * 2, positions[i][1])
            layout.addWidget(label, positions[i][0] * 2 + 1, positions[i][1])
            self.labels[sensor] = label

        main_layout.addLayout(layout)
        self.setLayout(main_layout)
        self.setWindowTitle("Surround View")
        self.update_surround_view()

    def update_surround_view(self, sample_token=None, current_frame=None):
        if sample_token:
            self.sample_token = sample_token
        if current_frame is not None:
            self.current_frame = current_frame
            self.frame_info_label.setText(f"Frame {self.current_frame + 1} / {self.total_frames}")

        for sensor, label in self.labels.items():
            img = self.loaded_images[self.sample_token].get(sensor, Image.new('RGB', (100, 100)))
            qt_img = pil2pixmap(img).scaled(300, 200, Qt.KeepAspectRatio)
            label.setPixmap(qt_img)
