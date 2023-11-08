import cv2


def main():
    # Initializing the source:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Initializing the models:
    face_detector = detector_model()
    landmarks_detector = landmark_model()

    while True:
        # Acquire the frame:
        _, frame = cap.read()

        # The detection works in grayscale:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces:
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        # Detect landmarks:
        try:
            _, landmarks= landmarks_detector.fit(gray, faces)
        except cv2.error:
            landmarks = [None]*len(faces)

        # Work on every face:
        for box, landmark  in zip(faces, landmarks):
            if landmark is not None:
                # Draw the landmarks:
                for x,y in landmark[0]:
                    x, y = round(x), round(y)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), 2)

            # Draw the rectangle on the frame:
            cv2.rectangle(frame, box[0:2], box[0:2] + box[2:], (0, 0, 255), 2)

        # Showing the image in a window:
        cv2.imshow('', frame)

        # Managing keys:
        c = cv2.waitKey(1)
        if c == 27:
            break

    # Releasing resources:
    cap.release()
    cv2.destroyAllWindows()


def detector_model():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
    )


def landmark_model():
    landmark_detector = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel('models/lbfmodel.yaml')
    return landmark_detector


if __name__ == "__main__":
    main()
