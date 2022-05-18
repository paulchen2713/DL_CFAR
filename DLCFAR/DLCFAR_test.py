from keras.models import load_model
import scipy.io as sio
import numpy as np
from timeit import default_timer as timer

NUM_SAMPLE_test=200000

mat = sio.loadmat('testing_noT_truncated_20_H1_SNR-4.mat')
# mat = sio.loadmat('testing_H3_SNR-4.mat')
x_test = mat['RDmap_input_raw_truncated']  # array
x_test = x_test.reshape(NUM_SAMPLE_test, 16, 16, 1).astype('float32')
model = load_model('DLCFAR_trained_weights_final_1.h5')

start = timer()  # 開始計時
y_test_hat = model.predict(x_test)
end = timer()
print('time consume:%.3f s ' % (end - start))
print('time average:', (end - start)/NUM_SAMPLE_test)

filename = "testing_DLCFAR_noise.csv"
np.savetxt(filename, y_test_hat, delimiter=",")
