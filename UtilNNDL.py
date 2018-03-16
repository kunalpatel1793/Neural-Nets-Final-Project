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

from sklearn.metrics import confusion_matrix
import itertools

import pandas as pd
import scipy
import scipy.signal

#Augments the dataset by creating windowed data samples 
def create_window_data(data, labels, windows=10, window_size=512, time_last = True ):
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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
 
'''
Formats the data so there are no nans and pust it into testing and training batches
Returns either arrays of all the data concatenated into four arrays: data and labels for test and train
OR returns four dicts where the Key "A0xT" (x = subject number) selects one of the 9 subjects, 
see bottom of the function for more details 
Sample usage:
file_path = '/home/carla/Downloads/project_datasets/project_datasets/'
train_data, test_data, train_labels, test_labels = prepare_data(file_path, 
                                                                num_test_samples = 50, 
                                                                verbose= False, 
                                                                return_all=True)
print train_data.shape
print train_labels.shape
print test_data.shape
print test_labels.shape

(2108, 22, 1000)
(2108, 4)
(450, 22, 1000)
(450, 4)
'''
def prepare_data(file_path, num_test_samples = 50, verbose= False, return_all=True, num_files = 9):
    train_data = {}
    test_data = {}
    train_labels = {}
    test_labels = {}
    for i in range(1,num_files+1):
        if verbose:
            print i
        A0it = h5py.File(file_path + 'A0{}T_slice.mat'.format(i),'r')
        data_i = np.copy(A0it['image'])[:,:22,:]
        labels_i = np.copy(A0it['type'])
        labels_i = labels_i[0,0:data_i.shape[0]:1]
        labels_i = np.asarray(labels_i, dtype=np.int32)
        labels_i = to_categorical(labels_i-769, num_classes=4)

        bad_indexes = np.unique(np.argwhere(np.isnan(data_i))[:,0])
        if verbose:
            print bad_indexes
            print data_i.shape
            print labels_i.shape
        data_i = np.delete(data_i,bad_indexes,0)
        labels_i = np.delete(labels_i,bad_indexes,0)
        if verbose:
            print data_i.shape
            print labels_i.shape

        train_data_i = data_i[:data_i.shape[0]-num_test_samples]
        test_data_i = data_i[data_i.shape[0]-num_test_samples:]
        train_labels_i = labels_i[:data_i.shape[0]-num_test_samples]
        test_labels_i = labels_i[data_i.shape[0]-num_test_samples:]
        
        train_data['A0{}T'.format(i)] = train_data_i
        test_data['A0{}T'.format(i)] = test_data_i
        train_labels['A0{}T'.format(i)] = train_labels_i
        test_labels['A0{}T'.format(i)] = test_labels_i
        
        if verbose:
            print train_data['A0{}T'.format(i)].shape
            print test_data['A0{}T'.format(i)].shape
            print train_labels['A0{}T'.format(i)].shape
            print test_labels['A0{}T'.format(i)].shape

        if i == 1:
            data_all = data_i
            labels_all = labels_i

            train_data_all = train_data_i
            test_data_all = test_data_i
            train_labels_all = train_labels_i
            test_labels_all = test_labels_i
        else:
            data_all = np.vstack([data_all, data_i]) 
            labels_all = np.concatenate([labels_all, labels_i])

            train_data_all = np.vstack([train_data_all, train_data_i])  
            test_data_all = np.vstack([test_data_all, test_data_i]) 
            train_labels_all = np.concatenate([train_labels_all, train_labels_i]) 
            test_labels_all = np.concatenate([test_labels_all, test_labels_i])
    if verbose:
        print data.keys()
        print labels.keys()
        print train_data_all.shape
        print train_labels_all.shape
    
    if return_all:
        #returns all the data concatenated 
        return train_data_all, test_data_all, train_labels_all, test_labels_all
    else:
        #returns all the data in dict format
        return train_data, test_data, train_labels, test_labels

def exponential_running_standardize(data, factor_new=0.001,
                                    init_block_size=None, eps=1e-4):
    """
    Perform exponential running standardization. 
    
    Compute the exponental running mean :math:`m_t` at time `t` as 
    :math:`m_t=\mathrm{factornew} \cdot mean(x_t) + (1 - \mathrm{factornew}) \cdot m_{t-1}`.
    
    Then, compute exponential running variance :math:`v_t` at time `t` as 
    :math:`v_t=\mathrm{factornew} \cdot (m_t - x_t)^2 + (1 - \mathrm{factornew}) \cdot v_{t-1}`.
    
    Finally, standardize the data point :math:`x_t` at time `t` as:
    :math:`x'_t=(x_t - m_t) / max(\sqrt{v_t}, eps)`.
    
    
    Parameters
    ----------
    data: 2darray (time, channels)
    factor_new: float
    init_block_size: int
        Standardize data before to this index with regular standardization. 
    eps: float
        Stabilizer for division by zero variance.
    Returns
    -------
    standardized: 2darray (time, channels)
        Standardized data.
    """
    df = pd.DataFrame(data)
    meaned = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    squared = demeaned * demeaned
    square_ewmed = squared.ewm(alpha=factor_new).mean()
    standardized = demeaned / np.maximum(eps, np.sqrt(np.array(square_ewmed)))
    standardized = np.array(standardized)
    if init_block_size is not None:
        other_axis = tuple(range(1, len(data.shape)))
        init_mean = np.mean(data[0:init_block_size], axis=other_axis,
                            keepdims=True)
        init_std = np.std(data[0:init_block_size], axis=other_axis,
                          keepdims=True)
        init_block_standardized = (data[0:init_block_size] - init_mean) / \
                                  np.maximum(eps, init_std)
        standardized[0:init_block_size] = init_block_standardized
    return standardized

def bandpass_cnt(data, low_cut_hz, high_cut_hz, fs, filt_order=3, axis=0):
    """
     Bandpass signal applying **causal** butterworth filter of given order.
    Parameters
    ----------
    data: 2d-array
        Time x channels
    low_cut_hz: float
    high_cut_hz: float
    fs: float
    filt_order: int
    Returns
    -------
    bandpassed_data: 2d-array
        Data after applying bandpass filter.
    """
    if (low_cut_hz == 0 or low_cut_hz is None) and (
                    high_cut_hz == None or high_cut_hz == fs / 2.0):
        log.info("Not doing any bandpass, since low 0 or None and "
                 "high None or nyquist frequency")
        return data.copy()
    if low_cut_hz == 0 or low_cut_hz == None:
        log.info("Using lowpass filter since low cut hz is 0 or None")
        return lowpass_cnt(data, high_cut_hz, fs, filt_order=filt_order, axis=axis)
    if high_cut_hz == None or high_cut_hz == (fs / 2.0):
        log.info(
            "Using highpass filter since high cut hz is None or nyquist freq")
        return highpass_cnt(data, low_cut_hz, fs, filt_order=filt_order, axis=axis)

    nyq_freq = 0.5 * fs
    low = low_cut_hz / nyq_freq
    high = high_cut_hz / nyq_freq
    b, a = scipy.signal.butter(filt_order, [low, high], btype='bandpass')
    assert filter_is_stable(a), "Filter should be stable..."
    data_bandpassed = scipy.signal.lfilter(b, a, data, axis=axis)
    return data_bandpassed


def filter_is_stable(a):
    """
    Check if filter coefficients of IIR filter are stable.
    
    Parameters
    ----------
    a: list or 1darray of number
        Denominator filter coefficients a.
    Returns
    -------
    is_stable: bool
        Filter is stable or not.  
    Notes
    ----
    Filter is stable if absolute value of all  roots is smaller than 1,
    see [1]_.
    
    References
    ----------
    .. [1] HYRY, "SciPy 'lfilter' returns only NaNs" StackOverflow,
       http://stackoverflow.com/a/8812737/1469195
    """
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a)))
    # from http://stackoverflow.com/a/8812737/1469195
    return np.all(np.abs(np.roots(a))<1)