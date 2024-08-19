import os
import numpy as np
import cv2


def data_loader(input_dir, categories):
    data = []
    labels = []
    for category_idx, category in enumerate(categories):
        for file in os.listdir(os.path.join(input_dir, category)):
            img_path = os.path.join(input_dir, category, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            data.append(img.flatten())
            labels.append(category_idx)

    data = np.asarray(data)
    labels = np.asarray(labels)
    return data, labels


x, y = data_loader('data', ['cat', 'dog'])

print('data', x)
print('target', y)