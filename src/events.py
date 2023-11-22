from __future__ import annotations
import cv2
import numpy as np
from .shapes import Rect, Landmarks
from .tools import flatten
from pyee.asyncio import AsyncIOEventEmitter
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial


ee = AsyncIOEventEmitter()


async def main():
    ctx = {}
    while cv2.waitKey(1) != 27:
        await source({"ctx": ctx, "shapes": {}})


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
    ee.emit('image', frame)


@ee.on('image')
async def preprocess(frame):
    gray = await wrap(cv2.cvtColor, frame["image"], cv2.COLOR_BGR2GRAY)
    frame["gray"] = gray
    ee.emit('gray', frame)


@ee.on('gray')
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
    ee.emit('faces', frame)


@ee.on('faces')
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
    ee.emit('landmarks', frame)


@ee.on('landmarks')
async def display(frame):
    image = frame["image"]
    for shape in flatten(frame["shapes"].values()):
        shape.draw(image)
    cv2.imshow('', image)


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


if __name__ == "__main__":
    asyncio.run(main())
