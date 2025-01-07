import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
import numpy as np
import os
import argparse
from torchvision.models import resnet18, ResNet18_Weights

from dataloader import CatsAndDogsDataset
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description="dogs and cats classification")
    parser.add_argument("--data_path", "-d", type=str, default="datasource")
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--batch_size", "-b", type=int, default=16)
    parser.add_argument("--lr", "-lr", type=float, default=0.001)
    parser.add_argument("--resume", "-r", type=bool, default=False)
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="trained_models")
    parser.add_argument("--tensorboard_dir", "-t", type=str, default="animal_board")

    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap='cool')
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    cm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, cm[i, j], color=color, horizontalalignment='center')
    plt.tight_layout()
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    writer.add_figure("Confusion matrix", figure, epoch)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size))
    ])
    training_set = CatsAndDogsDataset(args.data_path, is_train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=training_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    val_set = CatsAndDogsDataset(args.data_path, is_train=False, transform=transform)
    val_dataloader = DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, "last.pt"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["best_acc"]
    else:
        start_epoch = 0
        best_acc = -1

    # set tensorboard
    if not os.path.isdir(args.tensorboard_dir):
        os.mkdir(args.tensorboard_dir)
    writer = SummaryWriter(args.tensorboard_dir)
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    num_step_per_epoch = len(train_dataloader)
    # train
    for epoch in range(start_epoch, args.epochs):
        """Train"""
        model.train()
        progress_bar = tqdm(train_dataloader, colour='cyan')
        losses = []
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            avg_loss = np.mean(losses)
            progress_bar.set_description(
                "Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, args.epochs, avg_loss)
            )
            writer.add_scalar(tag="Train/Loss", scalar_value=avg_loss, global_step=epoch*num_step_per_epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        """Validation"""
        model.eval()
        losses = []
        all_labels = []
        all_predictions = []
        progress_bar = tqdm(val_dataloader, colour="yellow")
        for images, labels in progress_bar:
            all_labels.extend(labels)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.tolist())
            loss = criterion(outputs, labels)
            losses.append(loss.item())
        avg_loss = np.mean(losses)
        avg_acc = accuracy_score(all_labels, all_predictions)
        print("Epoch {}/{}. Loss {:0.4f}. Accuracy {:0.4f}".format(epoch + 1, args.epochs, avg_loss, avg_acc))
        writer.add_scalar(tag="Val/Loss", scalar_value=avg_loss, global_step=epoch)
        writer.add_scalar(tag="Val/Acc", scalar_value=avg_acc, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), training_set.classes, epoch)

        checkpoint = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(args.checkpoint_dir, "last.pt"))
        if avg_acc > best_acc:
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, "best.pt"))
            best_acc = avg_acc


if __name__ == "__main__":
    args = get_args()
    train(args)