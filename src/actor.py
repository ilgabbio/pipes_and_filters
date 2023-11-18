from __future__ import annotations
import cv2
import numpy as np
from .shapes import Rect, Landmarks
from .tools import flatten
import pykka
from queue import Queue


def main():
    # The actor system (a chain in this case):
    Runner(
        Source, Preprocessor, FacesDetector, LandmarksDetector, Drawer, Queuer
    ).run()


class Runner:
    def __init__(self, *actors):
        self._actors = actors

    def run(self):
        actor_refs = [self._actors[-1].start()]
        for actor in reversed(self._actors[:-1]):
            actor_refs.append(actor.start(actor_refs[-1]))
        actor_refs.reverse()

        while cv2.waitKey(1) != 27:
            # In a real scenario, these can have different rates:
            actor_refs[0].tell({"shapes": {}})
            image = actor_refs[-1].ask(None)
            if image is not None:
                cv2.imshow('', image)

        for actor in actor_refs:
            actor.stop()
        cv2.destroyAllWindows()


class Source(pykka.ThreadingActor):
    def __init__(self, dest):
        super().__init__()
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise IOError("Cannot open webcam")
        self._dest = dest

    def on_receive(self, frame: dict):
        _, image = self._cap.read()
        frame["image"] = image
        self._dest.tell(frame)

    def stop(self):
        super().stop()
        self._cap.release()


class Preprocessor(pykka.ThreadingActor):
    def __init__(self, dest):
        super().__init__()
        self._dest = dest

    def on_receive(self, frame):
        gray = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2GRAY)
        frame["gray"] = gray
        self._dest.tell(frame)


class FacesDetector(pykka.ThreadingActor):
    def __init__(self, dest):
        super().__init__()
        self._detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        )
        self._dest = dest

    def on_receive(self, frame):
        faces = tuple(map(Rect,
            self._detector.detectMultiScale(frame["gray"], 1.1, 4)
        ))
        frame["shapes"]["faces"] = faces
        self._dest.tell(frame)


class LandmarksDetector(pykka.ThreadingActor):
    def __init__(self, dest) -> None:
        super().__init__()
        self._detector = cv2.face.createFacemarkLBF()
        self._detector.loadModel('models/lbfmodel.yaml')
        self._dest = dest

    def on_receive(self, frame):
        faces = frame["shapes"]["faces"]
        try:
            _, lands = self._detector.fit(frame["gray"], np.array([f.rect for f in faces]))
        except cv2.error:
            lands = [None] * len(faces)
        frame["shapes"]["landmarks"] = tuple(map(Landmarks, lands))
        self._dest.tell(frame)


class Drawer(pykka.ThreadingActor):
    def __init__(self, dest) -> None:
        super().__init__()
        self._dest = dest

    def on_receive(self, frame):
        image = frame["image"]
        for shape in flatten(frame["shapes"].values()):
            shape.draw(image)
        self._dest.tell(frame)


class Queuer(pykka.ThreadingActor):
    def __init__(self, queue: Queue = None):
        super().__init__()
        self._queue = Queue(maxsize=10) if queue is None else queue

    def on_receive(self, frame: dict | None):
        try:
            if frame is not None:
                self._queue.put_nowait(frame["image"])
            else:
                return self._queue.get_nowait()
        except:
            return None


if __name__ == "__main__":
    main()
