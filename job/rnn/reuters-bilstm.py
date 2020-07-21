# Dataset: reuters
# Model: biLSTM
# Reference: 

# Import packages
from datetime import datetime
import math
import time
import pickle
import argparse
import tensorflow as tf

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
    print('Cannot found GPU. Training with CPU')
else:
    print('Found GPU at :{}'.format(device_name))

# Get arguments for job
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--prof_point', default=1.5, type=float)
parser.add_argument('--prof_or_latency', default='profiling', type=str)
parser.add_argument('--optimizer', default='Adadelta', type=str)
args = parser.parse_args()

max_features = 20000
maxlen = 128
num_data = 11228
num_classes = 46

batch_size = args.batch_size
prof_point = args.prof_point
batch_num = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_point)
prof_start = math.floor(batch_num * prof_point)
prof_len = 1
prof_range = '{}, {}'.format(prof_start, prof_start + prof_len)
prof_or_latency = args.prof_or_latency
optimizer = args.optimizer

# Get train/test dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=max_features, test_split=0.2)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Build biLSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 64, input_length=maxlen))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# Setting for tensorboard profiling callback
job_name = "reuters-bilstm"
logs = "/home/ubuntu/Deep-Cloud/logs/" + "{}-{}-{}-{}".format(job_name, optimizer, batch_size, datetime.now().strftime("%Y%m%d-%H%M%S"))
# logs = "/home/ubuntu/Deep-Cloud/logs/"  + str(batch_size) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = prof_range)

# Setting for latency check callback
class BatchTimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.all_times = []

    def on_train_end(self, logs=None):
        time_filename = "/home/ubuntu/Deep-Cloud/tensorstats/times-" + "{}-{}-{}-{}.pickle".format(job_name, optimizer, batch_size, datetime.now().strftime("%Y%m%d-%H%M%S"))
        time_file = open(time_filename, 'ab')
        pickle.dump(self.all_times, time_file)
        time_file.close()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_times = []
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_time_end = time.time()
        self.all_times.append(self.epoch_time_end - self.epoch_time_start)
        self.all_times.append(self.epoch_times)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_time_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_times.append(time.time() - self.batch_time_start)

latency_callback = BatchTimeCallback()

if prof_or_latency == 'profiling':
    # Start training with profiling
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks = [tboard_callback])
elif prof_or_latency == 'latency':
    # Start training with check latency
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks = [latency_callback])
else:
    print('error')