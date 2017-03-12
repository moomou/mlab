#!/usr/bin/env python
import numpy as np

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization

from keras.layers import (
    Input,
    Activation,
    Dense,
    LSTM,
    Dropout,
)
from keras.utils.np_utils import to_categorical

from keras.callbacks import ModelCheckpoint

from data import open_files, SEQ_LEN

WORD_EMBEDDING_DIM = 50

def build_model():
    # define the model
    model = Sequential()
    # TODO: play with memory unit count
    model.add(LSTM(512, return_sequences=True, input_shape=(SEQ_LEN, WORD_EMBEDDING_DIM)))

    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(LSTM(512, return_sequences=True))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(LSTM(512))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Dense(15, activation='softmax'))

    # output dimension should be the same as the input

    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['precision'])
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())
    input()

    f_test_y, f_test_X, f_train_y, f_train_X = open_files()

    test_y = to_categorical(np.array(f_test_y['data']), 15)
    test_X = np.array(f_test_X['data'])

    train_y = to_categorical(np.array(f_train_y['data']), 15)
    train_X = np.array(f_train_X['data'])

    callbacks = [
        ModelCheckpoint(
            filepath='./checkpoints/plang-{epoch:02d}-{val_loss:.2f}.hdf5',
            period=3
        )
    ]

    model.fit(
        train_X,
        train_y,
        validation_data=(test_X, test_y),
        nb_epoch=50,
        batch_size=64,
        callbacks=callbacks,
    )
