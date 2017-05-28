import re
import sys
import os

from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization, Activation
from keras.layers import (
    TimeDistributed,
    concatenate,
    Embedding
)
from keras.optimizers import Adam
import keras
import keras.backend as K
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.callbacks

from util import DEFAULT_CHAR_SET

CHAR_MAX_LEN = 512
CHAR_SET_SIZE = len(DEFAULT_CHAR_SET)


def conv_pool(model, filters=64, kernel_size=7, pool=True):
    model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                     strides=1, activation='relu', padding='causal'))
    model.add(keras.layers.advanced_activations.ELU())
    model.add(BatchNormalization())

    if pool:
        model.add(MaxPooling1D(pool_size=3))


def create_model(nb_classes):
    model = Sequential()
    model.add(Embedding(CHAR_SET_SIZE, 64, input_length=CHAR_MAX_LEN))

    conv_pool(model, filters=128)
    conv_pool(model, filters=128)
    conv_pool(model, kernel_size=3, pool=False)
    conv_pool(model, kernel_size=3, pool=False)
    conv_pool(model, kernel_size=3, pool=False)
    conv_pool(model, kernel_size=3)

    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='causal'))
    model.add(keras.layers.advanced_activations.ELU())
    model.add(GlobalMaxPool1D())

    model.add(Dense(CHAR_MAX_LEN * 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(CHAR_MAX_LEN * 2, activation='relu'))

    model.add(Dense(nb_classes, activation='softmax'))

    return model
