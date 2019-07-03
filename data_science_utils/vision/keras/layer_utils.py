
import keras
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from IPython.display import display
import seaborn as sns
from .visualize_layer import visualize_layer
from .adabound import AdaBound
from .one_cycle_lr import OneCycleLR, LRFinder
from keras.datasets import cifar10

from keras import backend as K
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score, accuracy_score
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, DepthwiseConv2D, Conv2D, SeparableConv2D, AveragePooling2D
from keras.layers import Input, concatenate
from keras.layers import Activation, Flatten, Dense, Dropout, Lambda, SpatialDropout2D, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.regularizers import l2
from keras_contrib.callbacks import CyclicLR
from keras.models import Model
import tensorflow as tf
import random
def concat_s2d(inputs):
    final_inputs = None
    if type(inputs) != list:
        return inputs
    if type(inputs) == list and len(inputs) == 1:
        return inputs[0]
    if type(inputs) == list:
        modified_inputs = []
        mh, mw = 1e8, 1e8  # min height and width

        for inpt in inputs:
            s = K.int_shape(inpt)
            h, w = s[-3], s[-2]
            mh = min(mh, h)
            mw = min(mw, w)
            modified_inputs.append((inpt, h, w))
        final_inputs = []
        for inpt, h, w in modified_inputs:
            assert h % mh == 0 and w % mw == 0 and h / mh == w / mw
            if int(h / mh) > 1:
                inp = Lambda(lambda x: tf.space_to_depth(x, block_size=int(h / mh)))(inpt)
            else:
                inp = inpt
            final_inputs.append(inp)
    inputs = concatenate(final_inputs)
    return inputs


def transition_layer(inputs, name, n_kernels=32,bn=True):
    inputs = concat_s2d(inputs)
    out = Conv2D(n_kernels,
                 kernel_size=(1, 1),
                 strides=1,
                 padding='same',
                 kernel_regularizer=l2(1e-4),
                 dilation_rate=1,
                 name=name)(inputs)
    out = BatchNormalization()(out) if bn else out
    out = Activation("relu")(out)
    return out


def conv_layer(inputs, name, n_kernels=32, kernel_size=(3, 3), dropout=0.0, dilation_rate=1, padding='same',
               enable_transition=False, transition_layer_kernels=32, strides=1, spatial_dropout=0.0,bn=True,
               bn_zero_gamma=False):
    inputs = concat_s2d(inputs)
    inputs = transition_layer(inputs, name + "_tran", transition_layer_kernels) if enable_transition else inputs
    out = Conv2D(n_kernels,
                 kernel_size=kernel_size,
                 strides=strides,
                 padding=padding,
                 kernel_regularizer=l2(1e-4),
                 dilation_rate=dilation_rate,
                 name=name + "conv_")(inputs)
    if bn:
        out = BatchNormalization(name=name + "bn_", gamma_initializer='zeros')(
            out) if bn_zero_gamma else BatchNormalization(name=name + "bn_")(out)
    out = Activation("relu", name=name + "activation_")(out)
    out = Dropout(dropout, name=name + "dropout_")(out) if dropout > 0 else out
    out = SpatialDropout2D(spatial_dropout)(out) if spatial_dropout > 0 else out
    return out


def depthwise_conv_layer(inputs, name, n_kernels=32, kernel_size=(3, 3), dropout=0.0, dilation_rate=1, padding='same',
                         depth_multiplier=1,
                         enable_transition=False, transition_layer_kernels=32, strides=1, spatial_dropout=0.0,bn=True,
                         bn_zero_gamma=False):
    inputs = concat_s2d(inputs)
    inputs = transition_layer(inputs, name + "_tran", transition_layer_kernels) if enable_transition else inputs
    out = SeparableConv2D(n_kernels,
                          kernel_size=kernel_size,
                          strides=strides,
                          padding=padding,
                          kernel_regularizer=l2(1e-4),
                          dilation_rate=dilation_rate,
                          depth_multiplier=depth_multiplier,
                          name=name + "sep-conv_")(inputs)
    if bn:
        out = BatchNormalization(name=name + "bn_", gamma_initializer='zeros')(
            out) if bn_zero_gamma else BatchNormalization(name=name + "bn_")(out)
    out = Activation("relu", name=name + "activation_")(out)
    out = Dropout(dropout, name=name + "dropout_")(out) if dropout > 0 else out
    out = SpatialDropout2D(spatial_dropout)(out) if spatial_dropout > 0 else out
    return out


def channel_shuffle(inputs):
    inputs = concat_s2d(inputs)
    _, w, h, in_channels = K.int_shape(inputs)
    mod_max = 1
    while in_channels % mod_max == 0:
        mod_max *= 2
    mod_max /= 2
    mod_max = min(mod_max, 8)
    mod_max = int(mod_max)

    def shuffle_layer(x):
        nb_chan_per_grp = in_channels // mod_max
        x = K.reshape(x, (-1, w, h, nb_chan_per_grp, mod_max))
        x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # Transpose only grps and chs
        x = K.reshape(x, (-1, w, h, in_channels))
        return x

    inputs = Lambda(shuffle_layer)(inputs)
    return inputs


def prediction_smoothing_categorical_cross_entropy(samples, batch_size, rate=1.05, epochs_without_decay=5,
                                                   smoothing_threshold=0.01):
    data = {}
    data['epoch'] = 0
    data['steps'] = 0
    steps_per_epoch = int(np.ceil(samples / batch_size))
    data['ones'] = None
    def loss(y, y_pred):
        data['steps'] += 1
        data['epoch'] = int(data['steps'] / steps_per_epoch) + 1  # which epoch is running
        decay = pow(rate, max(1, data['epoch'] - epochs_without_decay))
        assert decay > 1
        if data['epoch'] > epochs_without_decay:
            data['ones'] = tf.ones_like(y_pred) if data['ones'] is None else data['ones']
            y_pred = K.switch(K.greater_equal(y_pred, 1 - smoothing_threshold), data['ones'], y_pred)
            y_pred = K.switch(K.greater_equal(y_pred, 0.5), K.pow(y_pred, 1 / decay), y_pred)
            y_pred = K.switch(K.less(y_pred, 0.5), K.pow(y_pred, decay), y_pred)

        return K.categorical_crossentropy(y, y_pred)

    return loss
