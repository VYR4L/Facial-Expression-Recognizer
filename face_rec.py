import cv2
import face_recognition


webcam = cv2.VideoCapture(0)


class Camera():
    def __init__(self):
        self.webcam = webcam
        self.face_locations = []
        self.face_encodings = []

    def get_frame(self):
        ret, frame = self.webcam.read()
        self.face_locations = face_recognition.face_locations(frame)
        self.face_encodings = face_recognition.face_encodings(frame, self.face_locations)
        return frame

    def __del__(self):
        self.webcam.release()


def turn_on():
    camera = Camera()
    while True:
        frame = camera.get_frame()
        for (top, right, bottom, left) in camera.face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

