# DL_CFAR Architecture
from functools import wraps, reduce
from keras import Input
from keras.layers import Conv2D, Add, Flatten, Dense
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

def compose(*funcs):  # model
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

# parameters, same settings as the paper below
#   C. -H. Lin, Y. -C. Lin, Y. Bai, W. -H. Chung, T. -S. Lee and H. Huttunen, 
#   "DL-CFAR: A Novel CFAR Target Detection Method Based on Deep Learning," 
#   2019 IEEE 90th Vehicular Technology Conference (VTC2019-Fall), 2019, p.1-6
EPOCHS = 500
BATCH_SIZE = 128
NUM_SAMPLE_train = 200000
NUM_SAMPLE_val = 60000

@wraps(Conv2D)
def DL_Conv2D(*args, **kwargs):
    conv_kwargs = {'padding': 'same'}
    conv_kwargs.update(kwargs)
    return Conv2D(*args, **conv_kwargs)

def Conv2D_BN_PReLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
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

input_ = Input(shape=(16, 16, 1))
output_ = DLCFAR_body(input_)
model = Model(inputs=input_, outputs=output_)

model.compile(optimizer=Adam(lr=5 * 1e-5), loss='mean_squared_error') # Compile the model (determine optimizer and loss)
model.summary()



# Using TensorFlow backend.

# 2022-05-18 13:47:09.658251: I C:\tf_jenkins\workspace\rel-win\M\windows\PY\36\tensorflow\core\platform\cpu_feature_guard.cc:140] 
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            (None, 16, 16, 1)    0                                            
# __________________________________________________________________________________________________
# conv2d_1 (Conv2D)               (None, 16, 16, 32)   288         input_1[0][0]                    
# __________________________________________________________________________________________________
# batch_normalization_1 (BatchNor (None, 16, 16, 32)   128         conv2d_1[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_1 (PReLU)               (None, 16, 16, 32)   8192        batch_normalization_1[0][0]      
# __________________________________________________________________________________________________
# conv2d_2 (Conv2D)               (None, 16, 16, 16)   4608        p_re_lu_1[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_2 (BatchNor (None, 16, 16, 16)   64          conv2d_2[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_2 (PReLU)               (None, 16, 16, 16)   4096        batch_normalization_2[0][0]      
# __________________________________________________________________________________________________
# conv2d_3 (Conv2D)               (None, 16, 16, 8)    1152        p_re_lu_2[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_3 (BatchNor (None, 16, 16, 8)    32          conv2d_3[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_3 (PReLU)               (None, 16, 16, 8)    2048        batch_normalization_3[0][0]      
# __________________________________________________________________________________________________
# conv2d_4 (Conv2D)               (None, 16, 16, 1)    72          p_re_lu_3[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_4 (BatchNor (None, 16, 16, 1)    4           conv2d_4[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_4 (PReLU)               (None, 16, 16, 1)    256         batch_normalization_4[0][0]      
# __________________________________________________________________________________________________
# batch_normalization_5 (BatchNor (None, 16, 16, 1)    4           p_re_lu_4[0][0]                  
# __________________________________________________________________________________________________
# add_1 (Add)                     (None, 16, 16, 1)    0           input_1[0][0]                    
#                                                                  batch_normalization_5[0][0]      
# __________________________________________________________________________________________________
# p_re_lu_5 (PReLU)               (None, 16, 16, 1)    256         add_1[0][0]                      
# __________________________________________________________________________________________________
# conv2d_5 (Conv2D)               (None, 16, 16, 32)   288         p_re_lu_5[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_6 (BatchNor (None, 16, 16, 32)   128         conv2d_5[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_6 (PReLU)               (None, 16, 16, 32)   8192        batch_normalization_6[0][0]      
# __________________________________________________________________________________________________
# conv2d_6 (Conv2D)               (None, 16, 16, 16)   4608        p_re_lu_6[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_7 (BatchNor (None, 16, 16, 16)   64          conv2d_6[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_7 (PReLU)               (None, 16, 16, 16)   4096        batch_normalization_7[0][0]      
# __________________________________________________________________________________________________
# conv2d_7 (Conv2D)               (None, 16, 16, 8)    1152        p_re_lu_7[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_8 (BatchNor (None, 16, 16, 8)    32          conv2d_7[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_8 (PReLU)               (None, 16, 16, 8)    2048        batch_normalization_8[0][0]      
# __________________________________________________________________________________________________
# conv2d_8 (Conv2D)               (None, 16, 16, 1)    72          p_re_lu_8[0][0]                  
# __________________________________________________________________________________________________
# batch_normalization_9 (BatchNor (None, 16, 16, 1)    4           conv2d_8[0][0]                   
# __________________________________________________________________________________________________
# p_re_lu_9 (PReLU)               (None, 16, 16, 1)    256         batch_normalization_9[0][0]      
# __________________________________________________________________________________________________
# batch_normalization_10 (BatchNo (None, 16, 16, 1)    4           p_re_lu_9[0][0]                  
# __________________________________________________________________________________________________
# add_2 (Add)                     (None, 16, 16, 1)    0           p_re_lu_5[0][0]                  
#                                                                  batch_normalization_10[0][0]     
# __________________________________________________________________________________________________
# p_re_lu_10 (PReLU)              (None, 16, 16, 1)    256         add_2[0][0]                      
# __________________________________________________________________________________________________
# flatten_1 (Flatten)             (None, 256)          0           p_re_lu_10[0][0]                 
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 512)          131584      flatten_1[0][0]                  
# __________________________________________________________________________________________________
# dense_2 (Dense)                 (None, 256)          131328      dense_1[0][0]                    
# ==================================================================================================
# Total params: 305,312
# Trainable params: 305,080
# Non-trainable params: 232
# __________________________________________________________________________________________________