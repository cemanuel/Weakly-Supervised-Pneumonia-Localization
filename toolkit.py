import pandas as pd
import cv2
import pydicom
from constants import *
from sklearn import model_selection
import numpy as np

def get_image_array(patientId, img_width=IMG_W, img_height=IMG_H, path=TRAIN_DIR):

    get_img_path = lambda patientId: TEST_DIR + patientId + ".dcm"
    img_path = get_img_path(patientId)
    data = pydicom.read_file(img_path)
    img_array = data.pixel_array
    img_array = cv2.resize(img_array,(256,256))

    return img_array


def train_test_split(df, x_label, y_label, test_size=0.30, seed=7, down_sample = True):
    x = df[x_label]
    y = df[y_label]
    
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = test_size, random_state = seed)

    if down_sample:
        num_samples = y_train.value_counts().min()
        down_sample_x = []
        down_sample_y = []
        idx_all = []
        for label in y_train.value_counts().index:
           y_sample = y_train[y==label]
           x_sample = x_train[y==label]
           #index_range = range(y_sample.shape[0])
           index_range = y_sample.index
           idx = np.random.choice(index_range, size=num_samples, replace=False)
           #down_sample_x.append(x_sample[idx])
           #down_sample_y.append(y_sample[idx])
           idx_all += list(idx)
        np.random.shuffle(idx_all)
        #x_train = pd.concat(down_sample_x)
        #y_train = pd.concat(down_sample_y)
        x_train = x_train.loc[idx_all]
        y_train = y_train[idx_all]
    return x_train, x_test, y_train, y_test
    
