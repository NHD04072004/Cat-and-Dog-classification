from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets, models
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torchvision.utils import make_grid
from tempfile import TemporaryDirectory
import time
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

data_dir = 'datasource'

data_transform = {
    'training_set': v2.Compose([
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_set': v2.Compose([
        v2.RandomResizedCrop(256),
        v2.CenterCrop(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transform[x]) for x in ['training_set', 'test_set']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=16, shuffle=True) for x in ['training_set', 'test_set']}
dataset_size = {x: len(image_datasets[x]) for x in ['training_set', 'test_set']}
class_names = image_datasets['training_set'].classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

inputs, classes = next(iter(dataloaders['training_set']))
out = make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])
# plt.show()

def train(model, criterion, optimizer, scheduler, num_epoch=5):
    tik = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_param_pth = os.path.join(tempdir, 'best.pt')
        torch.save(model.state_dict(), best_model_param_pth)

        best_acc = 0.0

        for epoch in range(num_epoch):
            print(f'Epoch {epoch}/{num_epoch - 1}')
            print('-' * 10)

            for phase in ['training_set', 'test_set']:
                if phase == 'training_set':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.cuda().to(device)
                    labels = labels.cuda().to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'training_set'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'training_set':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'training_set':
                    scheduler.step()

                epoch_loss = running_loss / dataset_size[phase]
                epoch_acc = running_corrects.double() / dataset_size[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'test_set' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_param_pth)

            print()

        tok = time.time() - tik

        print(f'Training complete in {tok // 60:.0f}m {tok % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_param_pth, weights_only=True))

    return model


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test_set']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


model_ft = models.resnet18(weights='IMAGENET1K_V1')
model_ft.fc = nn.Linear(model_ft.fc.in_features, 2)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7)

model_ft = train(model_ft, criterion, optimizer_ft, exp_lr_scheduler)

visualize_model(model_ft)