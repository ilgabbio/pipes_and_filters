import cv2
import numpy as np

from barfi import st_barfi, barfi_schemas
import streamlit as st

from tools import Input, Output, define_block
from shapes import Rect, Landmarks

# pipenv run streamlit run src/p_and_f.py


@define_block("Camera", Output("image"))
def camera():
    cap = cv2.VideoCapture(0)
    for _ in range(10):
        _, image = cap.read()
    cap.release()
    return image

@define_block("Rgb2Gray", Input("image"), Output("gray"))
def rgb2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

@define_block("FaceDetector", Input("gray"), Output("faces"))
def faces(gray):
    model = 'haarcascade_frontalface_default.xml'
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + model)
    try:
        res = detector.detectMultiScale(gray, 1.1, 4)
    except Exception as e:
        res = []
    return tuple(map(Rect, res))

@define_block("LandmarksDetector",
              Input("gray"),
              Input("faces"),
              Output("landmarks"),
)
def landmarks(gray, faces):
    detector = cv2.face.createFacemarkLBF()
    detector.loadModel('models/lbfmodel.yaml')
    try:
        _, lands = detector.fit(gray, np.array([f.rect for f in faces]))
    except cv2.error:
        lands = [None] * len(faces)
    return  tuple(map(Landmarks, lands))

@define_block("Concatenate", Input("list1"), Input("list2"), Output("list"))
def cons(l1, l2):
    return [] + list(l1) + list(l2)

@define_block("Draw image", Input("image"), Input("shapes"), Output("drawn image"))
def draw_image(image, shapes):
    for shape in shapes:
        shape.draw(image)
    return image

@define_block("Save image", Input("image"))
def save_image(image):
    cv2.imwrite('generated/frame.png', image)

image_holder = st.empty()

@define_block("Display image", Input("image"))
def display_image(image):
    image_holder.image(image)

blocks = [camera, rgb2gray, faces, landmarks, cons, draw_image, save_image, display_image]

load_schema = st.selectbox('Select a saved schema:', barfi_schemas())
compute_engine = st.checkbox('Activate barfi compute engine', value=False)
if barfi_result := st_barfi(
    base_blocks=blocks,
    load_schema=load_schema,
    compute_engine=compute_engine,
    key="face-lands",
): st.write(barfi_result)
