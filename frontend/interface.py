from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QTextEdit, QVBoxLayout, QFileDialog, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
import sys
import os
from pathlib import Path
import backend.face_rec as face_rec
import backend.image_rec as image_rec
import cv2

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Janela Principal da Aplicação")
        self.caminhoarq = ""

        layout = QVBoxLayout()

        # Botão para selecionar uma foto
        select_photo_button = QPushButton("Selecionar Foto")
        select_photo_button.clicked.connect(self.select_files)
        select_photo_button.setFixedSize(300, 100)
        layout.addWidget(select_photo_button)

        # Caixa de texto para exibir resultados
        self.textBox = QTextEdit()
        self.textBox.setFixedSize(300, 50)
        self.textBox.setReadOnly(True)
        layout.addWidget(self.textBox)
        
        layout2 = QHBoxLayout()

        # Label para exibir a câmera
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480) 
        layout2.addWidget(self.camera_label)
        layout2.addLayout(layout)
        self.setLayout(layout2)

        # Timer para atualizar o frame da câmera
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)  # Atualiza a cada 30ms (~33 FPS)

    def update_camera(self):
        frame = face_rec.capture_frame()
        if frame is not None:
            # Converte o frame para o formato necessário pelo PyQt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def select_files(self):
        arq = 'Imagens (*.png *.jpg *.jpeg *.bmp)'
        file_path, _ = QFileDialog.getOpenFileName(self, 'Selecione o arquivo', os.getcwd(), arq)
        if file_path:
            self.image_path = file_path
            self.process_image()

    def process_image(self):
        result = image_rec.image_emotion_detection(self.image_path)
        self.textBox.setText(result)

