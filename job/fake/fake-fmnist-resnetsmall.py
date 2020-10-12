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


###################### Load Real Dataset ######################
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

# Build ResNet-small model
def res_block(x,filter,stride,name):
    input = x
    if stride != 1:
        input = tf.keras.layers.Conv2D(filters=filter,kernel_size=1,strides=stride,name=name+'_pooling_conv')(input)
        input = tf.keras.layers.BatchNormalization(name=name+'_pooling_bn')(input)

    x = tf.keras.layers.Conv2D(filters=filter,kernel_size=1,strides=stride,padding='same',name=name+'_conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_bn1')(x)
    x = tf.nn.relu(x,name=name+'_relu1')

    x = tf.keras.layers.Conv2D(filters=filter,kernel_size=1,strides=1,padding='same',name=name+'_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=name+'_bn2')(x)
    x = tf.keras.layers.add([input,x],name=name+'_add')

    x = tf.nn.relu(x,name=name+'_relu2')
    return x

def model_builder(x,attention):
    
    x = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,activation='relu',padding='same',name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(name='conv1_bn')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same',name='conv1_max_pool')(x)

    x = res_block(x,64,2,'ResBlock21')
    x = res_block(x,64,1,'ResBlock22')
    x = res_block(x,128,2,'ResBlock31')
    x = res_block(x,128,1,'ResBlock32')

    x =tf.keras.layers.GlobalAveragePooling2D(name='GAP')(x) 
    pred = tf.keras.layers.Dense(num_classes,activation='softmax')(x)
    
    return pred

inputs = tf.keras.Input(shape=(img_rows,img_cols,img_channels))
pred_normal = model_builder(inputs,None)
model = tf.keras.Model(inputs=inputs,outputs=pred_normal)
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

job_name = "fake-fmnist-resnetsmall"
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