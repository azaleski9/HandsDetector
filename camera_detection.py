from __future__ import absolute_import, division, print_function

import tensorflow as tf
import cv2 as cv2
import numpy as np

alpha_slider_max = 100
title_window = 'Camera hand detection example'
prob_threshold = 0.8


class FrameFragment:
    class_type = None

    def __init__(self, image, x, y, width, height):
        self.image = image
        self.height = height
        self.width = width
        self.y = y
        self.x = x


def get_proper_resolution(IMG_ROW, IMG_COL, __grid_size):
    ROW_DIFF = IMG_ROW / __grid_size - int(IMG_ROW / __grid_size)
    COLL_DIFF = IMG_COL / __grid_size - int(IMG_COL / __grid_size)
    if ROW_DIFF != 0:
        i = __grid_size
        while True:
            if i > IMG_ROW:
                IMG_ROW = i
                break
            else:
                i += __grid_size
    if COLL_DIFF != 0:
        i = __grid_size
        while True:
            if i > IMG_COL:
                IMG_COL = i
                break
            else:
                i += __grid_size
    return int(IMG_ROW), int(IMG_COL)


def split_image(image, __grid_size):
    crop_list = []
    rows = int(image.shape[0] / __grid_size)
    colls = int(image.shape[1] / __grid_size)
    for i in range(rows):
        y2 = int(image.shape[0] / rows * (i + 1))
        y1 = y2 - grid_size
        for j in range(colls):
            x2 = int(image.shape[1] / colls * (j + 1))
            x1 = x2 - __grid_size
            temp = image[y1:y2, x1:x2]
            crop_list.append(FrameFragment(temp, x1, y1, __grid_size, __grid_size))
    return crop_list


def on_trackbar(val):
    global prob_threshold
    prob_threshold = val / alpha_slider_max

if __name__ == "__main__":
    model = tf.keras.models.load_model('hand_model.h5')
    grid_size = 200
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        height, width = get_proper_resolution(cap.get(4), cap.get(3), grid_size)
        cv2.namedWindow(title_window)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (width, height))

            brightness = 0
            contrast = 50
            img = np.int16(frame)
            img = img * (contrast / 127 + 1) - contrast + brightness
            img = np.clip(img, 0, 255)
            frame = np.uint8(img)

            crops = split_image(frame, grid_size)
            images = []
            for crop in crops:
                img = crop.image

                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = tf.cast(img, tf.float32)
                img = (img / 255)
                img = tf.image.resize(img, (96, 96))
                #  img = tf.reshape(img, [-1, grid_size, grid_size, 3])
                images.append(img)

            print(images[0].shape)
            images = tf.convert_to_tensor(images, np.float32)
            print(images.shape)
            predictions = model.predict(images)
            for i in range(len(predictions)):
                if predictions[i][0] >= prob_threshold:
                    cv2.rectangle(frame, (crops[i].x, crops[i].y),
                                  (crops[i].x + crops[i].width, crops[i].y + crops[i].height), (0, 255, 0), 1)

            cv2.imshow(title_window, frame)
            trackbar_name = 'Probability threshold x %d' % alpha_slider_max
            cv2.createTrackbar(trackbar_name, title_window, int(prob_threshold*100), alpha_slider_max, on_trackbar)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
