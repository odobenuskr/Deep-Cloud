# Dataset: babi
# Model: LSTM
# Reference: https://github.com/keras-team/keras/blob/master/examples/babi_rnn.py

# Import packages
from functools import reduce
import re
import tarfile
import numpy as np
import tensorflow as tf

# Check GPU Availability
device_name = tf.test.gpu_device_name()
if not device_name:
	raise SystemError('GPU Device Not Found')
print('Found GPU at :{}'.format(device_name))

# Get arguments for job
RNN = tf.keras.layers.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 20

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
    
challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
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

# Build LSTM model
x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

sentence = tf.keras.layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = tf.keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = RNN(SENT_HIDDEN_SIZE)(encoded_sentence)

question = tf.keras.layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = tf.keras.layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = RNN(QUERY_HIDDEN_SIZE)(encoded_question)

merged = tf.keras.layers.concatenate([encoded_sentence, encoded_question])
preds = tf.keras.layers.Dense(vocab_size, activation='softmax')(merged)

model = tf.keras.Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Setting for tensorboard profiling callback
logs = "/home/ubuntu/Deep-Cloud/logs/"  + str(args.batch_size) + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
prof_range = str(args.prof_start_batch) + ',' + str(args.prof_end_batch)
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = prof_range)

# Start training
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05,
          callbacks = [tboard_callback])