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


def main():
    ctx = {}
    while cv2.waitKey(1) != 27:
        source({"ctx": ctx, "shapes": {}})


def source(frame):
    # Example with context:
    cap = frame["ctx"].get("cap")
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam")
        frame["ctx"]["cap"] = cap

    _, image = cap.read()
    frame["image"] = image
    ee.emit('image', frame)


@ee.on('image')
def preprocess(frame):
    gray = cv2.cvtColor(frame["image"], cv2.COLOR_BGR2GRAY)
    frame["gray"] = gray
    ee.emit('gray', frame)


@ee.on('gray')
def faces_detector(frame):
    face_model = frame["ctx"].get("face_model")
    if face_model is None:
        face_model = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        )
        frame["ctx"]["face_moel"] = face_model

    faces = tuple(map(Rect,
        face_model.detectMultiScale(frame["gray"], 1.1, 4)
    ))
    frame["shapes"]["faces"] = faces
    ee.emit('faces', frame)


@ee.on('faces')
def faces_landmarks(frame):
    landmarks_model = frame["ctx"].get("landmarks_model")
    if landmarks_model is None:
        landmarks_model  = cv2.face.createFacemarkLBF()
        landmarks_model.loadModel('models/lbfmodel.yaml')
        frame["ctx"]["landmarks_model"] = landmarks_model

    faces = frame["shapes"]["faces"]
    try:
        _, lands = landmarks_model.fit( 
            frame["gray"],
            np.array([f.rect for f in faces]),
        )
    except cv2.error:
        lands = [None] * len(faces)
    frame["shapes"]["landmarks"] = tuple(map(Landmarks, lands))
    ee.emit('landmarks', frame)


@ee.on('landmarks')
def display(frame):
    image = frame["image"]
    for shape in flatten(frame["shapes"].values()):
        shape.draw(image)
    cv2.imshow('', image)


if __name__ == "__main__":
    main()
