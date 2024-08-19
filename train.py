import tensorflow as tf
from data_loader import data_loader
from sklearn.model_selection import train_test_split


data, labels = data_loader('data', ['cat', 'dog'])
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)

x = tf.keras.Sequential()
x.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), use_bias=False, input_shape=(128, 128, 1)))
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1)))


x.add(tf.keras.layers.Conv2D(64, (3, 3), (1, 1), use_bias=False))
x.add(tf.keras.layers.Activation('relu'))
x.add(tf.keras.layers.MaxPool2D((2, 2), strides=(1, 1)))

x.add(tf.keras.layers.Dense(1024, activation='relu'))
x.add(tf.keras.layers.Dropout(0.2))
x.add(tf.keras.layers.Dense(1, activation='sigmoid'))
x.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    metrics=['accuracy']
)
history = x.fit(x=x_train, y=y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))
print(history)
