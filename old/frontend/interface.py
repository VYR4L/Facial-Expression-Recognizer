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
    '''
    Janela principal da aplicação.
    Contém um botão para selecionar uma imagem e uma área de texto para exibir os resultados.
    Também exibe a câmera em tempo real e detecta emoções usando o modelo Xception treinado.

    A classe herda de QWidget e utiliza o PyQt6 para a interface gráfica.
    '''
    def __init__(self):
        '''
        Inicializa a janela principal da aplicação.
        Configura o layout, os botões e a câmera.
        Cria um timer para atualizar o frame da câmera a cada 30ms.
        '''
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
        '''
        Atualiza o frame da câmera a cada 30ms.
        Captura o frame da câmera e o exibe na QLabel.
        Se o frame for None, não faz nada.
        '''
        frame = face_rec.capture_frame()
        if frame is not None:
            # Converte o frame para o formato necessário pelo PyQt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

    def select_files(self):
        '''
        Abre um diálogo para selecionar um arquivo de imagem.
        O arquivo deve ser uma imagem no formato PNG, JPG, JPEG ou BMP.
        Se um arquivo for selecionado, o caminho do arquivo é armazenado e a função process_image é chamada.
        Se nenhum arquivo for selecionado, não faz nada.
        '''
        arq = 'Imagens (*.png *.jpg *.jpeg *.bmp)'
        file_path, _ = QFileDialog.getOpenFileName(self, 'Selecione o arquivo', os.getcwd(), arq)
        if file_path:
            self.image_path = file_path
            self.process_image()

    # Processa a imagem selecionada e exibe o resultado na caixa de texto.
    # A função image_emotion_detection do módulo image_rec é chamada para detectar a emoção na imagem.
    def process_image(self):
        result = image_rec.image_emotion_detection(self.image_path)
        self.textBox.setText(result)

