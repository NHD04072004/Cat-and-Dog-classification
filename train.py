import tensorflow as tf
from data_loader import load_data
from model import build_model
from arguments import get_args
import matplotlib.pyplot as plt


def plot_training(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('images/cats_dogs_cls.png')
    plt.show()


def train():
    args = get_args()
    train_generator, val_generator = load_data(args.train_dir, args.val_dir, args.img_size, args.batch_size)
    model = build_model((*args.img_size, 3))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // args.batch_size,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // args.batch_size
    )

    model.save(args.model_save_path)
    
    plot_training(history)


if __name__ == "__main__":
    train()