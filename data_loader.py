import os
import numpy as np
import cv2



def data_loader(path_dir):
    path_imgs = []
    for i in os.listdir(path_dir):
        print(i)
        for j in os.listdir(os.path.join(path_dir, i)):
            path_imgs.append(os.path.join(path_dir, i, j))



data_loader('data')