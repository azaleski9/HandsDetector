import cv2 as cv2


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


if __name__ == "__main__":
    grid_size = 96
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        height, width = get_proper_resolution(cap.get(4), cap.get(3), grid_size)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (width, height))
            print(frame.shape)
            crops = split_image(frame, grid_size)
            for crop in crops:
                cv2.rectangle(frame, (crop.x, crop.y), (crop.x + crop.width, crop.y + crop.height), (0, 255, 0), 1)
            cv2.imshow('image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
