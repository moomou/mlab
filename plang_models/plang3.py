import pandas as pd

from keras.models import Model
from keras.layers import Dense, Input, Dropout, MaxPooling1D, Conv1D, GlobalMaxPool1D
from keras.layers import LSTM, Lambda, Bidirectional, concatenate, BatchNormalization, Activation
from keras.layers import TimeDistributed, concatenate
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import tensorflow as tf
import re
import keras.callbacks
import sys
import os

from util import DEFAULT_CHAR_SET

'''
Model Description

    raw text input
    |
    v
    split into by 1024 char
    |
    v
    train an embedding layer of 8 dim
    |
    v
    then uses 1D temporal CNN and RNN to "learn"
'''
MAX_SENTENCE = 10
CHAR_MAX_LEN = 512
CHAR_EMBEDDING = 8

CHAR_SET_SIZE = len(DEFAULT_CHAR_SET)

def max_1d(X):
    return K.max(X, axis=1)

def one_hot(x, size=CHAR_SET_SIZE):
    return tf.to_float(
        tf.one_hot(x, size, on_value=1, off_value=1, axis=-1))

def one_hot_shape(in_shape):
    return in_shape[0], in_shape[1], CHAR_SET_SIZE

def char_block(
    in_layer,
    nb_filter=(64, 100),
    filter_len=(3, 3),
    subsample=(2, 1),
    pool_size=(2, 2)
):
    block = in_layer

    for i in range(len(nb_filter)):
        block = Conv1D(filters=nb_filter[i],
                kernel_size=filter_len[i],
                padding='valid',
                activation='relu',
                kernel_initializer='glorot_normal',
                strides=subsample[i])(block)

        block = MaxPooling1D(pool_size=pool_size[i])(block)

    return block

def create_model(nb_classes):
    doc = Input(shape=(MAX_SENTENCE, CHAR_MAX_LEN), dtype='int64')
    sent = Input(shape=(CHAR_MAX_LEN,), dtype='int64',)

    embedded = Lambda(one_hot, output_shape=one_hot_shape)(sent)
    embedded = char_block(embedded, [192, 320], filter_len=[7, 5], subsample=[1, 1], pool_size=[2, 2])

    lstm_sent = \
        Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(embedded)

    sent_encode = Dropout(0.3)(lstm_sent)

    encoder = Model(inputs=sent, outputs=sent_encode)
    print('Encoder Summary')
    encoder.summary()

    # TODO: figure out what time dist is
    encoded = TimeDistributed(encoder)(doc)

    lstm_doc = \
    	Bidirectional(LSTM(128, return_sequences=False, dropout=0.15, recurrent_dropout=0.15, implementation=0))(encoded)

    output = Dropout(0.3)(lstm_doc)
    output = Dense(128, activation='relu')(output)
    output = Dropout(0.3)(output)
    output = Dense(1, activation='relu')(output)

    return Model(inputs=doc, outputs=output)
