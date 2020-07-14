# Dataset: CIFAR-10
# Model: LeNet-5

# Import packages
from datetime import datetime
import math
import time
import pickle
import os
import argparse
import tensorflow as tf

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
	raise SystemError('GPU Device Not Found')
print('Found GPU at :{}'.format(device_name))

# Get arguments for job
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--prof_start_batch', default=500, type=int)
parser.add_argument('--prof_end_batch', default=520, type=int)
args = parser.parse_args()

num_classes = 10
num_data = 50000
img_rows, img_cols = 32, 32

batch_size = 128
prof_start_batch = args.prof_start_batch
prof_end_batch = args.prof_end_batch
batch_data = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_end_batch/batch_data)

# Get train/test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build LeNet-5 model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Setting for tensorboard profiling callback
logs = "/home/ubuntu/Deep-Cloud/logs/"  + str(args.batch_size) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
prof_range = str(args.prof_start_batch) + ',' + str(args.prof_end_batch)
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = prof_range)

# Start training
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks = [tboard_callback])