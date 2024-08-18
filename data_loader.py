import os
import numpy as np
import cv2


def process_data(img_path, target_size=(128, 128)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def img_loader(path_dir):
    path_imgs = []
    for img in os.listdir(path_dir):
        path_img = path_dir + '/' + img
        path_imgs.append(path_img)
    return path_imgs


def data_loader(path_dir):
    imgs = []
    for img in img_loader(path_dir):
        img = process_data(img)
        imgs.append(img)
    return np.vstack(imgs)

print(img_loader('data'))
# print(data_loader('data'))