import cv2
import sys
import inspect
import numpy as np
from src.shapes import Rect, Landmarks


# A mini-autowiring framework:


def resource(func):
    func.is_resource = True
    return func


class autowire:
    def __init__(self, module, sink):
        self._sink = sink
        self._resources = []
        self._ctx = {}
        self._filters = {}

        missing = [sink]
        while missing:
            filter = missing.pop()
            spec = inspect.getfullargspec(filter)
            for name in spec.args:
                missing.append(getattr(module, name))
            self._filters[filter.__name__] = (filter, spec)

    def _evaluate(self, filter_name, ctx, data):
        filter, spec = self._filters[filter_name]
        args = []
        for arg in spec.args:
            if arg in ctx:
                args.append(ctx[arg])
            elif arg in data:
                args.append(data[arg])
            else:
                value = self._evaluate(arg, ctx, data)
                args.append(value)

        value = filter(*args)
        if getattr(filter, 'is_resource', False):
            self._resources.append(value)
            value = next(value)
            ctx[filter_name] = value
        else:
            data[filter_name] = value
        return value

    def dispose(self):
        for res in self._resources:
            try:
                next(res)
            except StopIteration:
                pass

    def __call__(self):
        return self._evaluate(self._sink.__name__, self._ctx, {})


# Our example graph:


@resource
def camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    yield cap
    cap.release()


def image(camera):
    _, image = camera.read()
    return image


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


@resource
def face_detector():
    model = 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + model)
    yield detector


def faces(face_detector, gray):
    try:
        res = face_detector.detectMultiScale(gray, 1.1, 4)
    except Exception:
        res = []
    return tuple(map(Rect, res))


@resource
def landmarks_detector():
    detector = cv2.face.createFacemarkLBF()
    detector.loadModel('models/lbfmodel.yaml')
    yield detector


def landmarks(landmarks_detector, gray, faces):
    try:
        _, lands = landmarks_detector.fit(gray, np.array([f.rect for f in faces]))
    except cv2.error:
        lands = [None] * len(faces)
    return tuple(map(Landmarks, lands))


def shapes(faces, landmarks):
    return faces + landmarks


def drawn_image(image, shapes):
    for shape in shapes:
        shape.draw(image)
    return image


@resource
def window():
    yield ''
    cv2.destroyAllWindows()


def display(window, drawn_image):
    cv2.imshow(window, drawn_image)


def main():
    module = sys.modules[__name__]
    graph = autowire(module, display)
    while cv2.waitKey(1) != 27:
        graph()
    graph.dispose()


if __name__ == "__main__":
    main()
