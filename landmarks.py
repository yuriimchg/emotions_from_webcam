import os
import pickle

import numpy as np
import dlib
import cv2


landmark_path = 'face_landmarks.dat'
keypoints_path = 'xgb_with.pickle'
input_src = 'media/filthy.mp4' or 0

detector = dlib.get_frontal_face_detector()
capture = cv2.VideoCapture(input_src)
predictor = dlib.shape_predictor(landmark_path)
emotions_encoded = pickle.loads(open(keypoints_path, 'rb'))


def shape_to_array(shape, dtype="int", landmarks_count=68):
    point = np.zeros((landmarks_count, 2), dtype=dtype)

    for i in range(landmarks_count):
        point[i] = (shape.part(i).x, shape.part(i).y)

    return point


def get_normalized_coords(shape, w, h):
    face_landmarks = []
    x_c, y_c = np.mean(shape, axis=0)
    for x, y in shape:
        face_landmarks.append([(x_c - x) / w, (y_c - y) / h])
    return np.array(face_landmarks)


def get_radius_vector(shape, w, h):
    rad_vec = []
    x_c, y_c = np.mean(shape, axis=0)
    for x, y in shape:
        rad_vec.append(np.sqrt(np.square((x_c - x) / w) + np.square((y_c - y) / h)))
    return rad_vec


def restore_coordinates(norm_shape, x_c, y_c, w, h):
    shape = np.array(norm_shape).reshape(68,3)
    shape = np.delete(shape, obj=0, axis=1)

    shape[:, 0] = - x_c + w * shape[:, 0]
    shape[:, 1] = - y_c + h * shape[:, 1]
    return shape


def get_face_landmarks(img_array, predictor):
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_image, 1)

    for i, rect in enumerate(rects):

        face_detector = predictor(gray_image, rect)
        shape = shape_to_array(face_detector)
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
