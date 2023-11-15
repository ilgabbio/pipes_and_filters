from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import cv2

from epypes import compgraph
from epypes.pipeline import Pipeline, SourcePipeline, SinkPipeline
from epypes.queue import Queue

from src.shapes import Rect, Landmarks


def main():
    # Some callables need initialization:
    camera = Camera()
    faces_detector = FacesDetector()
    landmarks_detector = LandmarksDetector()

    # The computational graph is defined by means of function and their relations. 
    runner = create_runner(
    #runner = create_pipelines(
        {'scale_factor': 1.1, 'min_neighbours': 4},
        func(camera, 'image'),
        func(rgb2gray, 'gray', 'image'),
        func(faces_detector, 'faces', 'gray', 'scale_factor', 'min_neighbours'),
        func(landmarks_detector, 'landmarks', 'gray', 'faces'),
        func(cons, 'shapes', 'faces', 'landmarks'),
        func(draw_image, 'drawn_image', 'image', 'shapes'),
        func(display, '', 'drawn_image'),
    )

    # We can execute the runner and extract results:
    while cv2.waitKey(1) != 27:
        runner.run()

    # Done, releasing resources:
    camera.stop()
    cv2.destroyAllWindows()

@dataclass(frozen = True)
class Func:
    name: str
    callable: Callable
    result: str
    parameters: Tuple[str, ...]

def func(callable: Callable, result: str, *parameters: str):
    try:
        name = callable.__name__
    except:
        name = callable.__class__.__name__
    return Func(name, callable, result, parameters)

def create_runner(hparams, *funcs: Func):
    return compgraph.CompGraphRunner(create_graph(*funcs), hparams)

def create_pipelines(hparams, *funcs: Func):
    #  Composition of pipelines:
    q = Queue()
    source = create_source(q, None, funcs[0])
    sink = create_sink(q, hparams, *funcs[1:])

    # The sink pipeline should listen on the queue:
    sink.listen()

    # The runnable part is the source:
    return source

def create_source(queue: Queue, frozen_tokens, *funcs: Func):
    def camera_out(pipe):
        return pipe['image'],

    return SourcePipeline(
        "Camera source",
        create_graph(*funcs),
        q_out=queue,
        out_prep_func=camera_out,
        frozen_tokens=frozen_tokens,
    )

def create_sink(queue: Queue, frozen_tokens, *funcs: Func):
    def image_dispatcher(image):
        return {'image': image[0]}

    return SinkPipeline(
        "Image processing",
        create_graph(*funcs),
        q_in=queue,
        event_dispatcher=image_dispatcher,
        frozen_tokens=frozen_tokens,
    )

def create_graph(*funcs: Func):
    return compgraph.CompGraph(
        func_dict={f.name: f.callable for f in funcs},
        func_io={f.name: (f.parameters, f.result) for f in funcs},
    )


# Our functions represent processing nodes.


class Camera:
    def __init__(self):
        self._cap = cv2.VideoCapture(0)

    def __call__(self):
        _, image = self._cap.read()
        return image

    def stop(self):
        self._cap.release()

def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class FacesDetector:
    def __init__(self):
        model = 'haarcascade_frontalface_default.xml'
        self._detector = cv2.CascadeClassifier(cv2.data.haarcascades + model)

    def __call__(self, gray, scale_factor = 1.1, min_neighbours = 4):
        try:
            res = self._detector.detectMultiScale(gray, scale_factor, min_neighbours)
        except Exception as e:
            res = []
        return tuple(map(Rect, res))

class LandmarksDetector:
    def __init__(self):
        self._detector = cv2.face.createFacemarkLBF()
        self._detector.loadModel('models/lbfmodel.yaml')

    def __call__(self, gray, faces):
        try:
            _, lands = self._detector.fit(gray, np.array([f.rect for f in faces]))
        except cv2.error:
            lands = [None] * len(faces)
        return  tuple(map(Landmarks, lands))

def cons(l1, l2):
    return [] + list(l1) + list(l2)

def draw_image(image, shapes):
    for shape in shapes:
        shape.draw(image)
    return image

def display(image):
    # Note: cannot work on a thread different from main:
    cv2.imshow('', image)
    #print("drawn image produced")


if __name__ == "__main__":
    main()
