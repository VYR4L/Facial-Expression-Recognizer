
from frontend.interface import MainWindow
from PyQt6.QtWidgets import QApplication
from backend.training_xception import TrainXception, XceptionCNN, SeparableConv2d
import sys
import torch


torch.serialization.add_safe_globals({'SeparableConv2d': SeparableConv2d})


def main():
    app = QApplication(sys.argv)

    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()



