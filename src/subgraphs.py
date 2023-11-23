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
from collections import deque
from threading import Thread


class Graph:
    def __init__(
            self,
            *filters: Tuple[Callable, str | None],
        ) -> None:
        self._sources = []
        self._loop = asyncio.new_event_loop()
        self._emitter = AsyncIOEventEmitter(loop=self._loop)
        for i, (filter, listens) in enumerate(filters):
            emits = filters[i+1][1] if i < len(filters)-1 else None
            if isinstance(listens, str):
                self._connect(filter, listens, emits)
            else:
                self._sources.append((filter, listens, emits))

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

    def run(self, stop_condition, data_factory, new_thread = False):
        async def aloop():
            while not stop_condition():
                await self._astep(data_factory())
        
        if new_thread:
            Thread(
                target=lambda: self._loop.run_until_complete(aloop()),
                daemon=True,
            ).start()
        else:
            self._loop.run_until_complete(aloop())

    async def _astep(self, data):
        async def run_one(source, wait, emits):
            await source(data)
            if emits is not None:
                self._emitter.emit(emits, data)

        for s in self._sources:
            must_wait = s[1]
            if must_wait:
                await run_one(*self._sources[-1])
            else:
                asyncio.create_task(run_one(*s))


_POOL = ThreadPoolExecutor()
async def wrap(op, *args): 
    """
    Wraps a call to be awaitable using a thread.
    Useful only if all operations release the GIL.
    """
    try:
        return await (
            asyncio
                .get_event_loop()
                .run_in_executor(_POOL, partial(op, *args))
        )
    except RuntimeError:
        # Loop stopped:
        pass


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
    image = frame.get("image")
    if image is None:
        return
    for shape in flatten(frame["shapes"].values()):
        shape.draw(image)
    cv2.imshow('', image)


def push(queue: deque):
    async def pusher(data):
        queue.append(data)
    return pusher

def pull(queue: deque, is_stop_required=None, wait=True):
    async def puller(data):
        if not queue and not wait:
            return
        while not queue:
            if is_stop_required and is_stop_required():
                return
            await asyncio.sleep(.1)
        val = queue.pop()
        data.clear()
        data.update(val)
    return puller


def main():
    # Context data:
    ctx = {}
    must_stop = False
    is_stop_required = lambda: must_stop
    def check_stop():
        nonlocal must_stop
        return (must_stop := cv2.waitKey(1) == 27)

    # A pipeline with two graphs:
    q1 = deque(maxlen=5)
    q2 = deque(maxlen=5)
    g1 = Graph(
            (pull(q2, wait=False), False),
            (display, 'drawn'),
            (source, True),
            (preprocess, 'image'),
            (faces_detector, 'gray'),
            (push(q1), 'faces'),
        )
    g2 = Graph(
            (pull(q1), True),
            (faces_landmarks, 'faces'),
            (push(q2), 'landmarks'),
        )

    # This run cannot guarantee the frame ordering:
    g2.run(is_stop_required, lambda: {}, new_thread=True)
    g1.run(check_stop, lambda: {"ctx": ctx, "shapes": {}})

    # Brutal, stop to be revisited:
    raise SystemExit

if __name__ == "__main__":
    main()
