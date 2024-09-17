from backend.face_rec import turn_on
from backend.training import training
from frontend import interface # interface bonitinha = TODO


def main():
    if __name__ == '__main__':
        print("Welcome to the AI Training and Face Recognition System!")
        print("Did you already train the AI? (y/n)")
        answer = input()
        if answer == 'n':
            training()
        else:
            ...
        turn_on()

    
main()
