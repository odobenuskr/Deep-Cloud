# Dataset: babi
# Model: RNN
# Reference: https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py

# Import packages
from functools import reduce
from datetime import datetime
import math
import time
import pickle
import argparse
import re
import tarfile
import numpy as np
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

# max_features = 20000
# maxlen = 256
# num_classes = 1

EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
num_data = 10000

batch_size = args.batch_size
prof_point = args.prof_point
batch_num = math.ceil(num_data/batch_size)
epochs = math.ceil(prof_point)
prof_start = math.floor(batch_num * prof_point)
prof_len = 1
prof_range = '{}, {}'.format(prof_start, prof_start + prof_len)
prof_or_latency = args.prof_or_latency
optimizer = args.optimizer

# Define functions for preprocessing
def tokenize(sent):
    return [x.strip() for x in re.split(r'(\W+)', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data

def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (tf.keras.preprocessing.sequence.pad_sequences(xs, maxlen=story_maxlen),
            tf.keras.preprocessing.sequence.pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))

# Get train/test dataset
try:
    path = tf.keras.utils.get_file('babi-tasks-v1-2.tar.gz',
                                              origin='https://s3.amazonaws.com/text-datasets/'
                                              'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
    
challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
with tarfile.open(path) as tar:
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train + test:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

# Build Simple RNN model
x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

sentence = tf.keras.layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = tf.keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = tf.keras.layers.SimpleRNN(SENT_HIDDEN_SIZE)(encoded_sentence)

question = tf.keras.layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = tf.keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = tf.keras.layers.SimpleRNN(QUERY_HIDDEN_SIZE)(encoded_question)

merged = tf.keras.layers.concatenate([encoded_sentence, encoded_question])
preds = tf.keras.layers.Dense(vocab_size, activation='softmax')(merged)

model = tf.keras.Model([sentence, question], preds)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Setting for tensorboard profiling callback
job_name = "babi-rnn"
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
        time_filename = "/home/ubuntu/Deep-Cloud/tensorstats/times-" + str(batch_size) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pickle"
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
    model.fit([x, xq], y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[tboard_callback])
elif prof_or_latency == 'latency':
    # Start training with check latency
    model.fit([x, xq], y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[latency_callback])
else:
    print('error')