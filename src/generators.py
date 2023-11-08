import cv2
import numpy as np
from .shapes import Rect, Landmarks
from .tools import flatten, compose


def main():
    # Initializing the pipeline:
    pipeline = compose(preprocessor, faces_detector, landmarks_detector, display)

    # Running the pipeline:
    for _ in pipeline(source()):
        pass


def source():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while cv2.waitKey(1) != 27:
        _, image = cap.read()
        yield {"image": image, "shapes": {}}

    cap.release()


def preprocessor(source):
    for frame in source:
        gray = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2GRAY)
        frame["gray"] = gray
        yield frame


def faces_detector(source):
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    )

    for frame in source:
        faces = tuple(map(Rect,
            detector.detectMultiScale(frame["gray"], 1.1, 4)
        ))
        frame["shapes"]["faces"] = faces
        yield frame


def landmarks_detector(source):
    detector = cv2.face.createFacemarkLBF()
    detector.loadModel('models/lbfmodel.yaml')

    for frame in source:
        faces = frame["shapes"]["faces"]
        try:
            _, lands = detector.fit(frame["gray"], np.array([f.rect for f in faces]))
        except cv2.error:
            lands = [None] * len(faces)
        frame["shapes"]["landmarks"] = tuple(map(Landmarks, lands))
        yield frame


def display(source):
    for frame in source:
        image = frame["image"]
        for shape in flatten(frame["shapes"].values()):
            shape.draw(image)
        cv2.imshow('', image)
        yield

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
