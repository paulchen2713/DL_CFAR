# import ipykernel
from functools import wraps
import numpy as np
import scipy.io as sio
import tensorflow as tf
from keras import Input
from keras.layers import Conv2D, Add, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from utils import compose
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, ModelCheckpoint

# Parameter
EPOCHS = 500
BATCH_SIZE = 128
NUM_SAMPLE_train = 200000
NUM_SAMPLE_val = 60000


@wraps(Conv2D)
def DL_Conv2D(*args, **kwargs):
    conv_kwargs = {'padding': 'same'}                          # padding的模式
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **conv_kwargs)


def Conv2D_BN_PReLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}                          # 默認不適用bias
    no_bias_kwargs.update(kwargs)
    return compose(
        DL_Conv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        PReLU())


def residual_block(x, num_filters):
    y = compose(
        Conv2D_BN_PReLU(num_filters * 32, (3, 3)),
        Conv2D_BN_PReLU(num_filters * 16, (3, 3)),
        Conv2D_BN_PReLU(num_filters * 8, (3, 3)),
        Conv2D_BN_PReLU(num_filters, (3, 3)),
        BatchNormalization())(x)
    x = Add()([x, y])
    return x


def DLCFAR_body(x):
    x = residual_block(x, 1)
    x = PReLU()(x)
    x = residual_block(x, 1)
    x = PReLU()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    return x


input = Input(shape=(16, 16, 1))
output = DLCFAR_body(input)
model = Model(inputs=input, outputs=output)

# Compile the model (determine optimizer and loss)
model.compile(optimizer=Adam(lr=5 * 1e-5), loss='mean_squared_error')
model.summary()

# =============================================================================
# tf.keras.utils.plot_model(model, to_file='DLCFAR_model.png', show_shapes=True)
# # input
# mat = sio.loadmat('training_input.mat')
# x_train = mat['RD_map_input_train']  # array
# mat = sio.loadmat('validation_input.mat')
# x_val = mat['RD_map_input_val']  # array
# # label
# mat = sio.loadmat('training_label.mat')
# y_train = mat['RD_map_label_train']  # array
# mat = sio.loadmat('validation_label.mat')
# y_val = mat['RD_map_label_val']  # array
# 
# x_train = x_train.reshape(NUM_SAMPLE_train, 16, 16, 1).astype('float32')
# x_val = x_val.reshape(NUM_SAMPLE_val, 16, 16, 1).astype('float32')
# y_train = y_train.astype('float32')
# y_val = y_val.astype('float32')
# 
# log_dir = 'logs/'  # 訓練過程及結果暫存路徑
# 
# logging = TensorBoard(log_dir=log_dir)
# # period:每隔3個epoch儲存一次，只儲存權重
# checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
#                               monitor='val_loss', save_weights_only=True, save_best_only=True,
#                               period=3)  # 訓練過程權重檔名稱由第幾輪加上損失率為名稱
# 
# history = model.fit(x_train, y_train,
#                     batch_size=BATCH_SIZE,
#                     epochs=EPOCHS,
#                     verbose=1,
#                     validation_data=(x_val, y_val),
#                     shuffle=True,
#                     callbacks=[logging, checkpoint])
# model.save(log_dir + 'DLCFAR_trained_weights_final_1.h5')  # 儲存臨時權重檔案名稱
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()
# 
# =============================================================================
