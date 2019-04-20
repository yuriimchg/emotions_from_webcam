import os

import numpy as np
import dlib
import cv2


landmark_path = 'face_landmarks.dat'
img_path = os.path.join('media', 'sample1.png')
detector = dlib.get_frontal_face_detector()
capture = cv2.VideoCapture(0)
predictor = dlib.shape_predictor(landmark_path)


def shape_to_np(shape, dtype="int", landmarks_count=68):
    point = np.zeros((landmarks_count, 2), dtype=dtype)

    for i in range(landmarks_count):
        point[i] = (shape.part(i).x, shape.part(i).y)

    return point

def get_face_landmarks(img_array, predictor):
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_image, 1)

    for i, rect in enumerate(rects):

        face_detector = predictor(gray_image, rect)
        shape = shape_to_np(face_detector)
        x1, y1 = rect.left(), rect.top()
        x2, y2 = rect.right(), rect.bottom()

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
        for x, y in shape:
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)


while True:
    ret, image = capture.read()
    get_face_landmarks(image, predictor)


    cv2.imshow('Press "q" to close the window', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
