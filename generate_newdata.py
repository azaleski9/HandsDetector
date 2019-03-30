import os
from os import path

import cv2 as cv2
import numpy as np
import csv


class InputImage:
    def __init__(self, filename, width, height, class_name, xmin, ymin, xmax, ymax):
        self.ymax = ymax
        self.xmax = xmax
        self.ymin = ymin
        self.xmin = xmin
        self.class_name = class_name
        self.height = height
        self.width = width
        self.filename = filename


def get_data(__csv_file, __images_path):
    __inputImages = []
    with open(__csv_file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if row[0] != "filename":
                inputImages.append(
                    InputImage(__images_path + row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
    return __inputImages


# def make_new_data(inputImages, )

if __name__ == "__main__":
    csv_file = "images/test_labels.csv"
    images_path = "images/test/"
    new_images_path = "new_images"
    if not os.path.exists(new_images_path):
        os.mkdir(new_images_path)
        os.mkdir(new_images_path + "/images")
    inputImages = get_data(csv_file, images_path)
    path = inputImages[0].filename
    img = cv2.imread(path, 0)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
