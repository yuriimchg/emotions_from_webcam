import os

import numpy as np
import dlib
import cv2


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

landmark_path = 'face_landmarks.dat'
img_path = os.path.join('media', 'sample1.png')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)

image = cv2.imread(img_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray_image, 1)

for i, rect in enumerate(rects):

    shape = predictor(gray_image, rect)
    shape = shape_to_np(shape)
    x1, y1 = rect.left(), rect.top()
    x2, y2 = rect.right(), rect.bottom()
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)

    for x, y in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow('fuck you', image)
if cv2.waitKey(0) & 0xFF == ord('z'):
    cv2.destroyAllWindows()
