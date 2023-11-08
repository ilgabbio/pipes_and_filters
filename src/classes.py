import cv2
from abc import ABC, abstractmethod
import numpy as np
from functools import reduce


def main():
    # Initializing the tools:
    source, preprocessor, face_detector, landmarks_detector, display, stopper = \
        (modules := (Source(), Preprocessor(), FacesDetector(), LandmarksDetector(), Display(), Stopper()))

    while not stopper():
        # Acquire the frame:
        frame = source()

        # The detection works in grayscale:
        gray = preprocessor(frame)

        # Detect faces:
        faces = face_detector(gray)

        # Detect landmarks:
        landmarks = landmarks_detector(gray, faces)

        # Showing the image in a window with the metadata:
        display(frame, [faces, landmarks])

    # Releasing resources:
    for module in modules:
        module.close()

class Source:
    def __init__(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise IOError("Cannot open webcam")

    def __call__(self):
        _, frame = self._cap.read()
        return frame

    def close(self):
        self._cap.release()


class Preprocessor:
    def __call__(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def close(self):
        pass


class Shape(ABC):
    @abstractmethod
    def draw(self, image):
        pass


class Rect:
    def __init__(self, xywh):
        self._xywh = xywh

    def draw(self, image):
        box = self._xywh
        cv2.rectangle(image, box[0:2], box[0:2] + box[2:], (0, 0, 255), 2)

    @property
    def rect(self):
        return self._xywh


class Landmarks:
    def __init__(self, coords=None):
        self._coords = coords[0]

    def draw(self, image):
        if self._coords is None:
            return

        for x,y in self._coords:
            x, y = round(x), round(y)
            cv2.circle(image, (x, y), 1, (0, 255, 0), 2)


class FacesDetector:
    def __init__(self):
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        )

    def __call__(self, image):
        return tuple(map(Rect,
            self._detector.detectMultiScale(image, 1.1, 4)
        ))

    def close(self):
        pass


class LandmarksDetector:
    def __init__(self):
        self._detector = cv2.face.createFacemarkLBF()
        self._detector.loadModel('models/lbfmodel.yaml')

    def __call__(self, image, faces):
        try:
            _, lands = self._detector.fit(image, np.array([f.rect for f in faces]))
        except cv2.error:
            lands = [None] * len(faces)
        return tuple(map(Landmarks, lands))

    def close(self):
        pass


class Display:
    def __call__(self, image, metadata=()):
        # Draw metadata:
        for shape in flatten(metadata):
            shape.draw(image)

        # Display:
        cv2.imshow('', image)

    def close(self):
        cv2.destroyAllWindows()


class Stopper:
    def __call__(self):
        return cv2.waitKey(1) == 27

    def close(self):
        pass


flat_map = lambda f, xs: reduce(lambda a, b: a + b, map(f, xs))
flatten = lambda xs: flat_map(lambda x: x, xs)


if __name__ == "__main__":
    main()
