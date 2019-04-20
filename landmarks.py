import os

import numpy as np
import dlib
import cv2


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

landmark_path = ''
img_path = os.path.join('media', 'sample1.png')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_path)

image = cv2.imread(img_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BRG2GRAY)

rects = detector(gray_image, 1)

for i, rect in enumerate(rects):

    shape = predictor(gray_image, rect)
    x, y = rect.left(), rect.top()
    w, h = rect.right() - x, rect.bottom() - y
