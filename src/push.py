from __future__ import annotations
import cv2
import numpy as np
from .shapes import Rect, Landmarks
from .tools import flatten
from abc import ABC, abstractmethod


def main():
    # Initializing and running the pipeline:
    Runner(
        Source() | Preprocessor() | FacesDetector() | LandmarksDetector() | Display()
    ).run()


class Runner:
    def __init__(self, source: Step):
        self._source = source

    def run(self):
        while not self._must_stop():
            self._source.push({"shapes": {}})
        self._source.push(None)

    def _must_stop(self) -> bool:
        return cv2.waitKey(1) == 27



class Step(ABC):
    def __init__(self):
        self._next: Step | None = None
    
    def push(self, whiteboard: dict | None):
        self.close() if whiteboard is None else self._op(whiteboard)

        if self._next is not None:
            self._next.push(whiteboard)

    def then(self, step: Step) -> Step:
        if self._next is not None:
            self._next.then(step)
        else:
            self._next = step
        return self

    def __or__(self, other: Step) -> Step:
        return self.then(other)
    
    @abstractmethod
    def _op(self, frame: dict) -> dict | None:
        pass

    def close(self):
        pass


class Source(Step):
    def __init__(self):
        super().__init__()
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise IOError("Cannot open webcam")

    def _op(self, frame: dict):
        _, image = self._cap.read()
        frame["image"] = image

    def close(self):
        self._cap.release()


class Preprocessor(Step):
    def _op(self, frame):
        gray = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2GRAY)
        frame["gray"] = gray


class FacesDetector(Step):
    def __init__(self):
        super().__init__()
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        )

    def _op(self, frame):
        faces = tuple(map(Rect,
            self._detector.detectMultiScale(frame["gray"], 1.1, 4)
        ))
        frame["shapes"]["faces"] = faces


class LandmarksDetector(Step):
    def __init__(self) -> None:
        super().__init__()
        self._detector = cv2.face.createFacemarkLBF()
        self._detector.loadModel('models/lbfmodel.yaml')

    def _op(self, frame):
        faces = frame["shapes"]["faces"]
        try:
            _, lands = self._detector.fit(frame["gray"], np.array([f.rect for f in faces]))
        except cv2.error:
            lands = [None] * len(faces)
        frame["shapes"]["landmarks"] = tuple(map(Landmarks, lands))


class Display(Step):
    def _op(self, frame):
        image = frame["image"]
        for shape in flatten(frame["shapes"].values()):
            shape.draw(image)
        cv2.imshow('', image)

    def close(self):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
