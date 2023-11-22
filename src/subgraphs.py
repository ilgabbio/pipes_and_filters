from __future__ import annotations
import cv2
import numpy as np
from typing import Callable, Tuple
from .shapes import Rect, Landmarks
from .tools import flatten
from pyee.asyncio import AsyncIOEventEmitter
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

_POOL = ThreadPoolExecutor()


async def wrap(op, *args): 
    """
    Wraps a call to be awaitable using a thread.
    Useful only if all operations release the GIL.
    """
    return await (
       asyncio
       .get_event_loop()
       .run_in_executor(_POOL, partial(op, *args))
   )


class Graph:
    def __init__(
            self,
            *filters: Tuple[Callable, str | None],
        ) -> None:
        self._sources = []
        self._emitter = AsyncIOEventEmitter()
        for i, (filter, listens) in enumerate(filters):
            emits = filters[i+1][1] if i < len(filters)-1 else None
            if listens is None:
                self._sources.append((filter,emits))
            else:
                self._connect(filter, listens, emits)

    def _connect(
            self,
            func: Callable,
            listens: str,
            emits: str | None,
        ):
        async def run_process(data):
            await func(data)
            if emits is not None:
                self._emitter.emit(emits, data)

        self._emitter.on(listens, run_process)

    def run(self, stop_condition, data_factory):
        async def aloop():
            while not stop_condition():
                await self._astep(data_factory())
        
        asyncio.run(aloop())

    async def _astep(self, data):
        for source, emits in self._sources:
            await source(data)
            if emits is not None:
                self._emitter.emit(emits, data)


async def source(frame):
    # Example with context:
    cap = frame["ctx"].get("cap")
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        frame["ctx"]["cap"] = cap

    _, image = await wrap(cap.read)
    frame["image"] = image


async def preprocess(frame):
    gray = await wrap(cv2.cvtColor, frame["image"], cv2.COLOR_BGR2GRAY)
    frame["gray"] = gray


async def faces_detector(frame):
    face_model = frame["ctx"].get("face_model")
    if face_model is None:
        face_model = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        )
        frame["ctx"]["face_moel"] = face_model

    faces = tuple(map(Rect,
        await wrap(face_model.detectMultiScale, frame["gray"], 1.1, 4)
    ))
    frame["shapes"]["faces"] = faces


async def faces_landmarks(frame):
    landmarks_model = frame["ctx"].get("landmarks_model")
    if landmarks_model is None:
        landmarks_model  = cv2.face.createFacemarkLBF()
        landmarks_model.loadModel('models/lbfmodel.yaml')
        frame["ctx"]["landmarks_model"] = landmarks_model

    faces = frame["shapes"]["faces"]
    try:
        _, lands = await wrap(
            landmarks_model.fit, 
            frame["gray"],
            np.array([f.rect for f in faces]),
        )
    except cv2.error:
        lands = [None] * len(faces)
    frame["shapes"]["landmarks"] = tuple(map(Landmarks, lands))


async def display(frame):
    image = frame["image"]
    for shape in flatten(frame["shapes"].values()):
        shape.draw(image)
    cv2.imshow('', image)


def main():
    # Creating a graph for asynchronous processing:
    graph = Graph(
        (source, None),
        (preprocess, 'image'),
        (faces_detector, 'gray'),
        (faces_landmarks, 'faces'),
        (display, 'landmarks'),
    )

    # This run cannot guarantee the frame ordering:
    ctx = {}
    graph.run(
        lambda: cv2.waitKey(1) == 27,
        lambda: {"ctx": ctx, "shapes": {}}
    )

if __name__ == "__main__":
    main()
