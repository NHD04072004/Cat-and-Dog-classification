import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a model to classify cats and dogs.")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--img_size', type=tuple, default=(128, 128))
    parser.add_argument('--train_dir', type=str, default='datasets/train')
    parser.add_argument('--val_dir', type=str, default='datasets/val')
    parser.add_argument('--model_save_path', type=str, default='model/cats_dogs_cls.h5')

    return parser.parse_args()
