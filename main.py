from backend.face_rec import turn_on, Camera
from backend.training import training_module, ResidualCNN, TrainModule
from frontend import interface # interface bonitinha = TODO
from pathlib import Path
import torch
import face_recognition
import cv2


def main():
    if __name__ == '__main__':
        print("Welcome to the AI Training and Face Recognition System!")
        print("Did you already train the AI? (y/n)")
        answer = input()
        if answer == 'n':
            training_module()
        elif answer == 'y':
            # Opening training file
            ROOT_DIR = Path(__file__).parent
            TRAINING_FILE = ROOT_DIR / 'model.pth'

            model = TrainModule()

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.load_state_dict(torch.load(TRAINING_FILE, map_location=device))

            model.to(device)
            model.eval()

            # Turn on the camera
            turn_on()
            

    
main()
