import tensorflow as tf
from data_loader import data_loader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


data_train, labels_train = data_loader('data/train', ['cats', 'dogs'])
data_test, labels_test = data_loader('data/test', ['cats', 'dogs'])

x = tf.keras.Sequential()
# x.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), input_shape=(130, 130, 1)))
# x.add(tf.keras.layers.BatchNormalization())
# x.add(tf.keras.layers.Activation('relu'))
# x.add(tf.keras.layers.MaxPooling2D((2, 2)))

# x.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1)))
# x.add(tf.keras.layers.BatchNormalization())
# x.add(tf.keras.layers.Activation('relu'))
# x.add(tf.keras.layers.MaxPooling2D((2, 2)))

# x.add(tf.keras.layers.Conv2D(128, (3, 3), (1, 1)))
# x.add(tf.keras.layers.BatchNormalization())
# x.add(tf.keras.layers.Activation('relu'))
# x.add(tf.keras.layers.MaxPooling2D((2, 2)))

x.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), input_shape=(128, 128, 1)))
x.add(tf.keras.layers.BatchNormalization())
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPooling2D((2, 2)))

x.add(tf.keras.layers.Conv2D(128, (3, 3), (1, 1)))
x.add(tf.keras.layers.BatchNormalization())
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPooling2D((2, 2)))

x.add(tf.keras.layers.Conv2D(256, (3, 3), (1, 1)))
x.add(tf.keras.layers.BatchNormalization())
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPooling2D((2, 2)))

x.add(tf.keras.layers.Conv2D(512, (3, 3), (1, 1)))
x.add(tf.keras.layers.BatchNormalization())
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPooling2D((2, 2)))

x.add(tf.keras.layers.Flatten())
x.add(tf.keras.layers.Dense(512, activation='relu'))
x.add(tf.keras.layers.Dropout(0.3))
# x.add(tf.keras.layers.Dense(32, activation='relu'))
# x.add(tf.keras.layers.Dropout(0.2))
x.add(tf.keras.layers.Dense(1, activation='sigmoid'))

x.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = x.fit(x=data_train, y=labels_train, epochs=100, validation_data=(data_test, labels_test), callbacks=[early_stopping])
test_loss, test_acc = x.evaluate(data_test, labels_test)
print(test_acc)
print(test_loss)

history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

x.save('models', 'cat_dog_cls.h5')


## plot
history_dict = history.history

plt.figure(figsize=(12, 5))

# accuracy
plt.subplot(1, 2, 1)
plt.plot(history_dict['accuracy'], label='Training Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# loss
plt.subplot(1, 2, 2)
plt.plot(history_dict['loss'], label='Training Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_history.png')

# plt.show()