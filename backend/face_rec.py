import cv2
from pathlib import Path
import torch
from backend.training_xception import TrainXception, SeparableConv2d
import torch.nn.functional as F
from torchvision import transforms

torch.serialization.add_safe_globals({'SeparableConv2d': SeparableConv2d})

# Carregar o modelo
model = TrainXception()

ROOT_DIR = Path(__file__).parent.parent
TRAINED_XCEPTION_MODEL = ROOT_DIR / 'training' / 'model_xception.pth'

model.load_state_dict(torch.load(TRAINED_XCEPTION_MODEL))
model.eval()

# Lista de emoções, definidas na mesma ordem do treinamento
emotions = ['Raiva', 'Desgosto', 'Medo', 'Feliz', 'Neutro', 'Triste', 'Surpresa']

# Transformações para pré-processamento da imagem
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Iniciar captura de vídeo pela webcam
cap = cv2.VideoCapture(0)

# Carregar o classificador de rosto pré-treinado do OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Função para capturar o frame e retornar o frame com a emoção detectada
def capture_frame():
    ret, frame = cap.read()
    if not ret:
        return None

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos no frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Desenhar um retângulo ao redor do rosto detectado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Extrair a Região de Interesse (ROI) do rosto e aplicar as transformações
        face_roi = gray[y:y+h, x:x+w]
        face_tensor = transform(face_roi).unsqueeze(0)

        # Fazer a predição da emoção
        with torch.no_grad():
            predicao = model(face_tensor)
            emotion_idx = torch.argmax(predicao, dim=1).item()
            emotion = emotions[emotion_idx]

        # Exibir a emoção detectada no frame
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame
