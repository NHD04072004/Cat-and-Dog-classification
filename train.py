import tensorflow as tf
from data_loader import data_loader


data_train, labels_train = data_loader('data/train', ['cats', 'dogs'])
data_test, labels_test = data_loader('data/test', ['cats', 'dogs'])

x = tf.keras.Sequential()
x.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), use_bias=False, input_shape=(128, 128, 1)))
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1)))

x.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), use_bias=False))
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1)))

x.add(tf.keras.layers.Dense(512, activation='relu'))
x.add(tf.keras.layers.Dropout(0.2))
x.add(tf.keras.layers.Dense(1, activation='sigmoid'))
x.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics=['accuracy']
)
history = x.fit(x=data_train, y=labels_train, batch_size=4, epochs=10, validation_data=(data_test, labels_test))
print(history)
