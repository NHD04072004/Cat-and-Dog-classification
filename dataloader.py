import os
import cv2
from torch.utils.data import Dataset, DataLoader

class CatsAndDogsDataset(Dataset):
    def __init__(self, path, is_train):
        self.classes = ['cats', 'dogs']
        if is_train:
            data_path = os.path.join(path, 'training_set')
        else:
            data_path = os.path.join(path, 'test_set')

        self.images = []
        self.labels = []

        for class_name in self.classes:
            sub_folder_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(sub_folder_path):
                image_path = os.path.join(sub_folder_path, image_name)
                self.images.extend(image_path)
                self.labels.extend(self.labels)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        return image, label


if __name__ == "__main__":
    dataset = CatsAndDogsDataset(path='datasource', is_train=True)
    image, label = dataset[100]

    print(image.shape)
    print(label)