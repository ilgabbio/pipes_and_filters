from __future__ import annotations
import cv2
import numpy as np
from .shapes import Rect, Landmarks
from .tools import flatten
from abc import ABC, abstractmethod
from streamz import Stream, gen
from tornado.ioloop import IOLoop


def main():
    # Creating the stream:
    source = Stream(asynchronous=True)
    
    # Time management is possible:
    camera = source.map(Camera()).rate_limit(1./10)

    # Complex graphs can be described (eventually with loops):
    face_detector = camera.map(preprocessor).map(FacesDetector())
    eyes_detector = face_detector.map(EyesDetector())
    landmarks_detector = face_detector.map(LandmarksDetector())

    # Many low-cost flow control operators, also custom (as 'merge'):
    eyes_detector.zip(landmarks_detector).merge().sink(display)

    # Can visualize the stream (using graphviz and networkx):
    source.visualize("generated/stream.png", rankdir="LR")    

    # Can be run in AsyncIO using Tornado:
    async def runner():
        while cv2.waitKey(1) != 27:
            await source.emit({"shapes": {}})

        # Poison pil:
        await source.emit(None)

    # Running the pipeline:
    IOLoop().run_sync(runner)


class Camera:
    def __init__(self):
        self.__name__ = self.__class__.__name__
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise IOError("Cannot open webcam")

    def __call__(self, frame: dict | None):
        if frame is None:
            self._cap.release()
        else:
            _, image = self._cap.read()
            frame["image"] = image
        return frame


def preprocessor(frame):
    if frame is not None:
        gray = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2GRAY)
        frame["gray"] = gray
    return frame


class Detector(ABC):
    def __init__(self):
        self.__name__ = self.__class__.__name__
        self._detector = cv2.CascadeClassifier(cv2.data.haarcascades + self.model)

    def __call__(self, frame):
        if frame is not None:
            self._op(frame)
        return frame

    @abstractmethod
    def _op(self, frame):
        pass

class FacesDetector(Detector):
    model = 'haarcascade_frontalface_default.xml'

    def _op(self, frame):
        faces = tuple(map(Rect,
            self._detector.detectMultiScale(frame["gray"], 1.1, 4)
        ))
        frame["shapes"]["faces"] = faces

class EyesDetector(Detector):
    model = 'haarcascade_eye_tree_eyeglasses.xml'

    def _op(self, frame):
        eyes = []
        for rect in frame["shapes"]["faces"]:
            x, y, w, h = rect.rect
            roi_gray = frame["gray"][y:y+h, x:x+w]
            eyes += [
                Rect(np.array(eye) + [x,y,0,0])
                for eye in self._detector.detectMultiScale(roi_gray)
            ]
        frame["shapes"]["eyes"] = eyes


class LandmarksDetector:
    def __init__(self) -> None:
        self.__name__ = self.__class__.__name__
        self._detector = cv2.face.createFacemarkLBF()
        self._detector.loadModel('models/lbfmodel.yaml')

    def __call__(self, frame):
        if frame is not None:
            faces = frame["shapes"]["faces"]
            try:
                _, lands = self._detector.fit(frame["gray"], np.array([f.rect for f in faces]))
            except cv2.error:
                lands = [None] * len(faces)
            frame["shapes"]["landmarks"] = tuple(map(Landmarks, lands))
        return frame


@Stream.register_api(attribute_name="merge")
class Merge(Stream):
    def __init__(self, upstream, **kwargs):
        super().__init__(upstream, ensure_io_loop=True, **kwargs)

    @gen.coroutine
    def update(self, x, who=None, metadata=None):
        yield self._emit(x[0], metadata=metadata)


def display(frame):
    if frame is None:
        cv2.destroyAllWindows()
    else:
        image = frame["image"]
        for shape in flatten(frame["shapes"].values()):
            shape.draw(image)
        cv2.imshow('', image)


if __name__ == "__main__":
    main()
