#File for storing utility functions for manipulation and visualization

import numpy as np
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import LSTM
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Convolution2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D

import kapre 
from kapre.time_frequency import Spectrogram

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime
now = datetime.now()
import librosa
from librosa import display

#Augments the dataset by creating windowed data samples 
def create_window_data(data, labels, windows=10, window_size=512):
    assert data.shape[0] == labels.shape[0]
    step = int(float(data.shape[2]-window_size)/float(windows-1))
    data_sliced = np.zeros([data.shape[0]*windows,data.shape[1],window_size])
    for t in range(data.shape[0]):
        for w in range(windows):
            data_slice = data[t,:,(w*step):window_size+(w*step)]
            data_sliced[(t*windows)+w] = data_slice

    labels_sliced  = np.repeat(labels, windows,axis=0)
    return data_sliced, labels_sliced

def plot_hist(vals,labels='Null',title='Null',xlabel='epochs'):
    n = len(vals)
    for i in range(n): plt.plot(vals[i],label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    plt.show()
