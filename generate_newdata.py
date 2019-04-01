# Example usage python generate_newdata.py images/test_labels.csv images/test/ new_images 80

import csv
import os
import sys

import cv2 as cv2
import numpy as np


class InputImage:
    def __init__(self, filename, width, height, class_name, xmin, ymin, xmax, ymax):
        self.ymax = int(ymax)
        self.xmax = int(xmax)
        self.ymin = int(ymin)
        self.xmin = int(xmin)
        self.class_name = class_name
        self.height = int(height)
        self.width = int(width)
        self.filename = filename


class OutputImage:
    def __init__(self, filename, class_name):
        self.class_name = class_name
        self.filename = filename

    def __iter__(self):
        return iter([self.filename, self.class_name])


class Rect:
    def __init__(self, x, y, width, height):
        self.height = height
        self.y = y
        self.x = x
        self.width = width
        self.area = (width * height)


def get_data(__csv_file, __images_path):
    __inputImages = []
    with open(__csv_file, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if row[0] != "filename":
                __inputImages.append(
                    InputImage(os.path.join(__images_path, row[0]), row[1], row[2], row[3], row[4], row[5], row[6], row[7]))
    return __inputImages


def resize_image(__image, __grid_size):
    IMG_ROW = __image.shape[0]
    IMG_COL = __image.shape[1]
    ROW_DIFF = IMG_ROW / __grid_size - int(IMG_ROW / __grid_size)
    COLL_DIFF = IMG_COL / __grid_size - int(IMG_COL / __grid_size)
    if ROW_DIFF != 0:
        IMG_ROW -= __grid_size * ROW_DIFF
    if COLL_DIFF != 0:
        IMG_COL -= __grid_size * COLL_DIFF
    return __image[0:int(IMG_ROW), 0:int(IMG_COL)], int(IMG_ROW / __grid_size), int(IMG_COL / __grid_size)


def check_collistion(rect1, rect2):
    if rect1.x < rect2.x + rect2.width and \
            rect1.x + rect1.width > rect2.x and \
            rect1.y < rect2.y + rect2.height and \
            rect1.y + rect1.height > rect2.y:
        return True
    return False


def crop_image(__imageFile, __image, __rows, __cols, path):
    images = []
    __grid_size = __image.shape[0] / __rows
    collider_counter = 0
    grid_area = __grid_size ** 2
    for i in range(__rows):
        y2 = int(__image.shape[0] / __rows * (i + 1))
        y1 = int(y2 - __grid_size)
        for j in range(__cols):
            x2 = int(__image.shape[1] / __cols * (j + 1))
            x1 = int(x2 - __grid_size)
            image = __image[y1:y2, x1:x2]
            _, filename = os.path.split(__imageFile.filename)
            filename = str(filename.split('.')[0]) + "_" + str(i) + "_" + str(j) + "_" + str(__imageFile.xmin) + ".jpg"
            file_path = os.path.join(path, "images", filename)

            # checking collision of cropped image with hand in original
            rect1 = Rect(x1, y1, (x2 - x1), (y2 - y1))
            rect2 = Rect(__imageFile.xmin, __imageFile.ymin, (__imageFile.xmax - __imageFile.xmin),
                         (__imageFile.ymax - __imageFile.ymin))

            if check_collistion(rect1, rect2):
                if rect1.x < rect2.x:
                    collision_width = (rect1.x + rect1.width) - rect2.x
                else:
                    collision_width = (rect2.x + rect2.width) - rect1.x
                if rect1.y < rect2.y:
                    collistion_height = (rect1.y + rect1.height) - rect2.y
                else:
                    collistion_height = (rect2.y + rect2.height) - rect1.y
                collistion_area = collistion_height * collision_width
                percentage_of_collision = collistion_area / grid_area
                if percentage_of_collision > 0.5:
                    images.append(OutputImage(filename, "hand"))
                    cv2.imwrite(file_path, image)
                    collider_counter += 1
                else:
                    if collider_counter > 0:
                        images.append(OutputImage(filename, "background"))
                        cv2.imwrite(file_path, image)
                        collider_counter -= 1
            else:
                if collider_counter > 0:
                    images.append(OutputImage(filename, "background"))
                    cv2.imwrite(file_path, image)
                    collider_counter -= 1
    return images


def make_new_data(__input_images, __path, __grid_size, __csv_filename):
    __csv_filename = "new_" + __csv_filename
    with open(os.path.join(__path, __csv_filename), 'w', newline='') as file:
        writter = csv.writer(file)
        i = 1
        for imageFile in __input_images:
            print(str(i) + " / " + str(len(__input_images)))
            i += 1
            image = cv2.imread(os.path.abspath(imageFile.filename))
            image, row_numbers, col_numbers = resize_image(image, __grid_size)
            images = crop_image(imageFile, image, row_numbers, col_numbers, __path)
           # for hand in images:
            #    if hand.class_name == "hand":
             #       temp2 = cv2.imread(os.path.join(__path, "images", hand.filename))
              #      cv2.imwrite(os.path.join(__path, "hands", hand.filename), temp2)
            for img in images:
                writter.writerow(list(img))
        file.close()


if __name__ == "__main__":
    # csv_file = "images/test_labels.csv"
    csv_file = sys.argv[1]
    # images_path = "images/test/"
    images_path = sys.argv[2]
    # new_images_path = "new_images"
    new_images_path = sys.argv[3]
    # grid_size = 80
    grid_size = int(sys.argv[4])
    _, csv_filename = os.path.split(csv_file)
    if not os.path.exists(new_images_path):
        os.mkdir(new_images_path)
    if not os.path.exists(os.path.join(new_images_path, "images")):
        os.mkdir(os.path.join(new_images_path, "images"))
    if not os.path.exists(os.path.join(new_images_path, "hands")):
        os.mkdir(os.path.join(new_images_path, "hands"))
    print("reading data")
    inputImages = get_data(csv_file, images_path)
    print("started making data")
    make_new_data(inputImages, new_images_path, grid_size, csv_filename)
    print("ended making data")
