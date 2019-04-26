import os
import pickle

import numpy as np
import dlib
import cv2


landmark_path = 'face_landmarks.dat'
keypoints_path = 'training/xgb_model.pickle'
input_src = 'media/ems.mp4' or 0
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
encoded_emotions = {t: em for t, em in enumerate(emotions)}

detector = dlib.get_frontal_face_detector()
capture = cv2.VideoCapture(input_src)
predictor = dlib.shape_predictor(landmark_path)
model = pickle.load(open(keypoints_path, "rb"))


def shape_to_array(shape, dtype="int", landmarks_count=68):
    point = np.zeros((landmarks_count, 2), dtype=dtype)

    for i in range(landmarks_count):
        point[i] = (shape.part(i).x, shape.part(i).y)

    return point


def get_normalized_coords(shape):
    face_landmarks = np.zeros(shape.shape)

    x_max = shape[:, 0].max()
    x_min = shape[:, 0].min()
    y_max = shape[:, 1].max()
    y_min = shape[:, 1].min()

    face_landmarks[:, 0] = (x_max - shape[:, 0]) / (x_max - x_min)
    face_landmarks[:, 1] = (y_max - shape[:, 1]) / (y_max - y_min)

    return face_landmarks


def get_radius_vector(norm_shape):
    x, y = np.split(np.array(norm_shape), 2, axis=1)
    x_c, y_c = np.mean(norm_shape, axis=0)
    return np.sqrt(np.square(x - x_c) + np.square(y - y_c))


def get_angle(norm_shape):
    x, y = np.split(np.array(norm_shape), 2, axis=1)
    return np.arctan(x / y)


def predict(model, norm_shape):
    return model.predict(norm_shape.reshape([1, 272]))


while True:
    ret, image = capture.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_image, 1)

    for i, rect in enumerate(rects):
        face_detector = predictor(gray_image, rect)
        shape = shape_to_array(face_detector)

        norm_shape = get_normalized_coords(shape)
        radius_vector = get_radius_vector(norm_shape)
        angle = get_angle(norm_shape)
        x, y = np.split(np.array(norm_shape), 2, axis=1)

        prediction = predict(model, np.array([x, y, radius_vector, angle]))

        x_c, y_c = np.mean(shape, axis=0)
        cv2.putText(image, f'{encoded_emotions[int(prediction)]}', (int(x_c), int(y_c)), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        for x, y in shape:
            cv2.circle(image, (x, y), 2, (255, 255, 255), -1)

    cv2.imshow('Press "q" to close the window', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
