import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from backend.training_xception import TrainXception, XceptionCNN, SeparableConv2d
from pathlib import Path


torch.serialization.add_safe_globals({'SeparableConv2d': SeparableConv2d})

# Definindo a arquitetura do modelo Xception
modelo = TrainXception()

# Carregando o modelo treinado
ROOT_DIR = Path(__file__).parent.parent
TRAINED_XCEPTION_MODEL = ROOT_DIR / 'training' / 'model_xception.pth'

modelo.load_state_dict(torch.load(TRAINED_XCEPTION_MODEL))
modelo.eval()

# Lista de emoções, definidas na mesma ordem do treinamento
emotions = ['Raiva', 'Desgosto', 'Medo','Felicidade', 'Neutro', 'Triste', 'Surpresa']

# Transformações para pré-processamento da imagem
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def image_emotion_detection(path_image):
    '''
    Função para detectar emoções em uma imagem usando o modelo Xception treinado.
    A imagem deve conter um rosto humano, e a função irá desenhar um retângulo ao redor do rosto e
    exibir a emoção detectada.

    params:
        path_image (str): Caminho para a imagem a ser analisada.
    returns:
        result (str): Emoção detectada na imagem.

    '''
    # Carregar a imagem
    image = cv2.imread(path_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Carregar o classificador de rosto do OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("Nenhum rosto detectado na imagem.")
        return

    for (x, y, w, h) in faces:
        # Desenhar um retângulo ao redor do rosto detectado
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Extrair e pré-processar a Região de Interesse (ROI) do rosto
        face_roi = gray[y:y+h, x:x+w]
        face_tensor = transform(face_roi).unsqueeze(0)

        # Fazer a predição da emoção
        with torch.no_grad():
            predicao = modelo(face_tensor)
            emotion_idx = torch.argmax(predicao, dim=1).item()
            emotion = emotions[emotion_idx]

        result = f'Emoção: {emotion}'
        return result
