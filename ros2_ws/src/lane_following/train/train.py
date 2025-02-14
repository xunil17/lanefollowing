import os
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Lambda, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from utils import load_multi_dataset, mkdir_p, HDF5_PATH, MODEL_PATH
from datetime import datetime
import time
from sklearn.model_selection import train_test_split

from models import build_nvidia_model, build_openpilot_model, build_modified_openpilot_model


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

print('Loading data from HDF5...')
X_data, Y_data = load_multi_dataset(os.path.join(HDF5_PATH, 'train_h5_list.txt'))
# X_test, Y_test = load_multi_dataset(os.path.join(HDF5_PATH, 'test_h5_list.txt'))

print('Number of images:', X_data.shape[0])
print('Number of labels:', Y_data.shape[0])

print('Splitting data into training set and testing set....')
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)

train_len = X_train.shape[0]
test_len = X_test.shape[0]

X_train_desire = np.tile(np.zeros(8), (train_len, 1))
X_train_rnn = np.tile(np.zeros(512), (train_len, 1))

X_test_desire = np.tile(np.zeros(8), (test_len, 1))
X_test_rnn = np.tile(np.zeros(512), (test_len, 1))

model = build_modified_openpilot_model()

model.summary()
model.compile(optimizer=Adam(lr=1e-04, decay=0.0), loss='mse')

t0 = time.time()
model.fit({'vision': X_train, 'desire': X_train_desire, 'rnn_state': X_train_rnn}, Y_train, validation_data=({'vision': X_test, 'desire': X_test_desire, 'rnn_state': X_test_rnn}, Y_test), shuffle=True, epochs=30, batch_size=128)
t1 = time.time()
print('Total training time:', t1 - t0, 'seconds')

mkdir_p(MODEL_PATH)
model_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_file = os.path.join(MODEL_PATH, '{}.h5'.format(model_id))
model.save(model_file)
print("Training done successfully and model has been saved: {}".format(model_file))
print("Drive safely!")
