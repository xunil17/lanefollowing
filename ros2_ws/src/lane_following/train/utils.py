import h5py
import numpy as np
import math
import cv2
import os
import errno

import numpy as np
from numba import jit, prange


BASE_PATH = os.path.dirname(os.path.realpath(__file__))
CSV_PATH = u'{}/data'.format(BASE_PATH)
IMG_PATH = u'{}/data/img'.format(BASE_PATH)
HDF5_PATH = u'{}/data/hdf5'.format(BASE_PATH)
MODEL_PATH = u'{}/model'.format(BASE_PATH)
# OPENPILOT_PATH = 
IMAGE_DIM = (256*2, 128*2)


def load_dataset(file_name):
	hdf5_path = '{}'.format(file_name)
	hdf5_file = h5py.File(hdf5_path, 'r')

	data_X = np.array(hdf5_file['data'][:])
	data_Y = np.array(hdf5_file['label'][:])

	return data_X, data_Y


def load_multi_dataset(txt_name):
    data_X, data_Y = None, None
    try:
        with open(txt_name, 'r') as f:
            for line in f:
                line = line.rstrip()
                x, y = load_dataset(line)
                if data_X is None or data_Y is None:
                    data_X, data_Y = x, y
                else:
                    data_X = np.concatenate((data_X, x))
                    data_Y = np.concatenate((data_Y, y))
    except FileNotFoundError as e:
        print(e)
        return data_X, data_Y

    return data_X, data_Y


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[0]
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    num_complete_minibatches = int(math.floor(m/mini_batch_size))
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def preprocess_image(cv_img, crop=False):
    if crop:
        cv_img = cv_img[540:960, :, :]

    cropped_img = cv_img[200:600, :]
    cv2.imshow("input", cropped_img)

    # 876 x 1164


    # 512 * 256
    cv_img = cv2.resize(cropped_img, IMAGE_DIM, interpolation=cv2.INTER_AREA)
    cv2.imshow("cropped", cv_img)
    cv2.waitKey(1)


    # print(cv_img.shape)
    cv_img = cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = rgb2yuv(cv_img)

    cv_img = np.transpose(cv_img, (2, 0, 1))
    # 6xwidthxheight
    # heightxwidth x 6

    # print(cv_img.shape)

    # cv_img = cv_img / 255. - 0.5

    return cv_img

@jit(nopython=True, parallel=True, nogil=True, debug=True)
def rgb2yuv(pix):
    img = np.zeros((pix.shape[0] // 2, pix.shape[1] // 2, 6))

    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            U_val = 0
            V_val = 0
            for k in prange(4):
                if k == 0:
                    p = pix[i * 2, j * 2]
                elif k == 1:
                    p = pix[i * 2, j * 2 + 1]
                elif k == 2:
                    p = pix[i * 2 + 1, j * 2]
                elif k == 3:
                    p = pix[i * 2 + 1, j * 2 + 1]

                Y_val = 0.25681631 * p[0] + 0.50415484 * p[1] + 0.09791402 * p[2]
                U = -0.14824553 * p[0] + -0.29102007 * p[1] + 0.4392656 * p[2]
                V = 0.43927107 * p[0] + -0.36783273 * p[1] - 0.07143833 * p[2]

                img[i, j, k] = (Y_val - 112) / 128
                U_val += U
                V_val += V

            img[i, j, 4] = U_val / 4 / 128
            img[i, j, 5] = V_val / 4 / 128

    return img

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    print('Directory is ready: {}'.format(path))


if __name__ == '__main__':
	load_multi_dataset('hdf5/train_h5_list.txt')
