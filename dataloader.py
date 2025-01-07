import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize

class CatsAndDogsDataset(Dataset):
    def __init__(self, path, is_train, transform):
        self.transform = transform
        self.classes = ['cats', 'dogs']
        if is_train:
            data_path = os.path.join(path, 'training_set')
        else:
            data_path = os.path.join(path, 'test_set')

        self.images = []
        self.labels = []

        for class_id, class_name in enumerate(self.classes):
            sub_folder_path = os.path.join(data_path, class_name)
            for image_name in os.listdir(sub_folder_path):
                image_path = os.path.join(sub_folder_path, image_name)
                self.images.append(image_path)
                self.labels.append(class_id)
                # print(self.images)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # image = cv2.imread(self.images[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]

        return image, label


if __name__ == "__main__":
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    dataset = CatsAndDogsDataset(path='datasource', is_train=True, transform=transform)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    for image, label in dataloader:
        print(image.shape, label)