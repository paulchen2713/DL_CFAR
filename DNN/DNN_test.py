 # -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:07:50 2020

@author: Yu
"""

from keras.models import load_model
import scipy.io as sio
import numpy as np

mat = sio.loadmat('DNN_testing_input_1.mat')
x_test = mat['DNN_input_raw']  # array
x_test = x_test.reshape(x_test.shape[0], 3, 3).astype('float32')

# mat = sio.loadmat('DNN_testing_input_label.mat')
# y_test = mat['DNN_input_label_raw']  # array
# y_test = y_test.astype('int64')

model = load_model('DNN_trained_weights_final.h5')

# Evaluate (test) the model

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('Test loss: {:.4f}'.format(test_loss))
# print('Test acc: {:.4f}'.format(test_acc))

y_test_hat = model.predict(x_test)

filename = "testing_DNN_finalresult.csv"
np.savetxt(filename, y_test_hat, delimiter=",")