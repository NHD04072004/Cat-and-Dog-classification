import os
import numpy as np
import cv2

for path_dir in os.listdir('data'):
    print(path_dir)
    for img in os.listdir(os.path.join('data' + '/' + path_dir)):
        path_img = os.path.join('data' + '/' + path_dir + '/' + img)
        read_img = cv2.imread(path_img)
        img_gray = cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY)
        img_resize = cv2.resize(img_gray, (128, 128))

        cv2.imshow('img', img_resize)
        cv2.waitKey(0)