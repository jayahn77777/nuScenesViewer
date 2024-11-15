import sys
from PyQt5.QtWidgets import QApplication
from nusc_map_viewer import NuScenesViewer

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = NuScenesViewer()
    sys.exit(app.exec_())
