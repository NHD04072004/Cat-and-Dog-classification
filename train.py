import tensorflow as tf


x = tf.keras.Sequential()
x.add(tf.keras.layers.Conv2D(32, (3, 3), (1, 1), use_bias=False))
x.add(tf.keras.layers.BatchNormalization())
x.add(tf.keras.layers.Activation('relu'))

x.add(tf.keras.layers.Conv2D(128, (3, 3), (1, 1), use_bias=False))
x.add(tf.keras.layers.BatchNormalization())
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
history = x.fit(x=None, batch_size=32, epochs=20, validation_data=None)
print(history)
