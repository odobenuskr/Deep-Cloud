# Dataset: reuters
# Model: SimpleRNN
# Reference: https://github.com/keras-team/keras/blob/master/examples/reuters_mlp.py

# Import packages
import numpy as np
import tensorflow as tf

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))

# Get arguments for job
max_features = 20000
max_words = 1000
maxlen = 256
batch_size = 128
epochs = 5
num_classes = 46

# Get train/test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=max_words, test_split=0.2)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 64))
model.add(tf.keras.layers.SimpleRNN(64, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(64, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(64, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(64))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Setting for tensorboard profiling callback
# logs = "/home/ubuntu/Deep-Cloud/logs/"  + str(args.batch_size) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
# prof_range = str(args.prof_start_batch) + ',' + str(args.prof_end_batch)
# tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                  histogram_freq = 1,
#                                                  profile_batch = prof_range)

# Start training
model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)