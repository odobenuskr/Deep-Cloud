from datetime import datetime
import math
import time
import pickle
import tensorflow as tf

import numpy as np

device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))

num_classes = 10
num_data = 60000
num_test = 10000
img_rows, img_cols, img_channels = 28, 28, 1
batch_size = 64
prof_point = 1.5
batch_num = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_point)
prof_start = math.floor(batch_num * prof_point)
prof_len = 1
prof_range = '{}, {}'.format(prof_start, prof_start + prof_len)
optimizer = 'Adadelta'


###################### Build Fake Dataset ######################
x_train_shape = (num_data, img_rows, img_cols, img_channels)
# x_train_shape = (60000, 28, 28, 1)
y_train_shape = (num_data, 1)
# y_train_shape = (60000, 1)

x_test_shape = (num_test, img_rows, img_cols, img_channels)
# x_test_shape = (10000, 28, 28, 1)
y_test_shape = (num_test, 1)
# y_test_shape = (10000, 1)

x_train = np.random.rand(*x_train_shape)
y_train = np.random.randint(num_classes, size=y_train_shape)
x_test = np.random.rand(*x_test_shape)
y_test = np.random.randint(num_classes, size=y_test_shape)
###############################################################


if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
    input_shape = (img_rows, img_cols, img_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

job_name = "fake-mnist-lenet5"
logs = "/home/ubuntu/Deep-Cloud/logs/" + "{}-{}-{}-{}".format(job_name, optimizer, batch_size, datetime.now().strftime("%Y%m%d-%H%M%S"))
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = prof_range)

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks = [tboard_callback])