import argparse
import cv2
from torchvision.models import resnet18, ResNet18_Weights
import os
import torch
import torch.nn as nn
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="dogs and cats classification")
    parser.add_argument("--image_path", "-p", type=str, required=True)
    parser.add_argument("--image_size", "-i", type=int, default=224)
    parser.add_argument("--checkpoint_dir", "-c", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, "best.pt"), weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    origin_image = cv2.imread(args.image_path)
    image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = image/255.
    image = np.transpose(image, (2, 0, 1))[None, :, :, :]
    image = torch.from_numpy(image)
    sigmoid = nn.Sigmoid()
    classes = ["Cat", "Dog"]

    with torch.no_grad():
        image = image.float().to(device)
        outputs = model(image)
        prediction = sigmoid(outputs)[0]
        cv2.imshow("Predicted: {}. Accuracy: {:0.2f}".format(classes[torch.argmax(prediction)], torch.max(prediction)), origin_image)
        cv2.waitKey(0)


if __name__ == "__main__":
    args = get_args()
    inference(args)