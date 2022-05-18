# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:49:44 2020

@author: Yu
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras import utils
import scipy.io as sio

log_dir = 'logs/'  # 訓練過程及結果暫存路徑

# input
mat = sio.loadmat('DNN_training_input.mat')
x_train = mat['DNN_input_raw']  # array
mat = sio.loadmat('DNN_val_input.mat')
x_val = mat['DNN_input_raw']  # array
# label
mat = sio.loadmat('DNN_training_input_label.mat')
y_train = mat['DNN_input_label_raw']  # array
mat = sio.loadmat('DNN_val_input_label.mat')
y_val = mat['DNN_input_label_raw']  # array

x_train = x_train.reshape(x_train.shape[0], 3, 3).astype('float32')
x_val = x_val.reshape(x_val.shape[0], 3, 3).astype('float32')
y_train = y_train.astype('int64')
y_val = y_val.astype('int64')

# Build the model using Sequential API
model = Sequential()
model.add(layers.Flatten(input_shape=(3,3))) # add the layers one-by-one
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(9, activation='sigmoid'))

# Inspect the model (check the parameters/shape/graph)
model.summary()
utils.plot_model(model, to_file='DNN_model.png', show_shapes=True)

# Compile the model (determine optimizer and loss)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

# Train the model
history = model.fit(x=x_train,
                    y=y_train,
                    validation_data=(x_val, y_val),
                    batch_size=128,
                    epochs=100)
model.save(log_dir + 'DNN_trained_weights_final.h5') 

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Training', 'Validation'])
plt.title('model accuracy')
