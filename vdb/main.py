#!/usr/bin/env python
import os
import time
from pprint import pformat

import keras
import h5py
import glog
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.objectives import (
    categorical_crossentropy,
    mean_squared_error,
    mean_absolute_error, )

from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger, )
from keras.optimizers import Adam, SGD, RMSprop
from keras.backend.tensorflow_backend import set_session
from tqdm import tqdm

from constant import (
    CHECKPT_DIR,
    SAMPLE_RATE,
    MFCC_SAMPLE_LEN_MS,
    MFCC_NB_COEFFICIENTS, )
from data_util import (
    DataMode,
    timit_h5_fname,
    vctk_h5_fname,
    fff_en_h5_fname,
    ffh_jp_h5_fname,
    voice_h5_fname, )
from model import (
    build_model,
    build_model2,
    build_model3,
    build_model4,
    build_model5,
    build_model6_softmax,
    load_model_h5,
    compute_receptive_field, )
from generator import one_hot
from pipeline import (
    get_speaker_generators,
    get_wav_generators,
    get_speaker_generators_softmax, )


def _set_tf_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.96
    set_session(tf.Session(config=config))


def categorical_mean_squared_error(y_true, y_pred):
    '''MSE for categorical variables.'''
    return K.mean(
        K.square(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)))


def sgd():
    return {
        'optimizer': 'sgd',
        'lr': 0.001,
        'momentum': 0.9,
        'decay': 0.,
        'clipnorm': 1.,
        'nesterov': True,
        'epsilon': None
    }


def adam():
    return {
        'momentum': 0.9,
        'optimizer': 'adam',
        'lr': 0.001,
        'decay': 0.,
        'epsilon': 1e-8
    }


def adam2():
    return {'optimizer': 'adam', 'lr': 0.015, 'decay': 1e-4, 'epsilon': 1e-10}


def make_optimizer(optimizer,
                   lr,
                   momentum=0.9,
                   decay=1e-6,
                   nesterov=True,
                   epsilon=1e-10,
                   **kwargs):
    if optimizer == 'sgd':
        optim = SGD(lr, momentum, decay, nesterov, **kwargs)
    elif optimizer == 'adam':
        optim = Adam(lr=lr, decay=decay, epsilon=epsilon, **kwargs)
    else:
        raise ValueError('Invalid config for optimizer.optimizer: ' +
                         optimizer)
    return optim


def _train3_setup(frame_length):
    _set_tf_session()

    input_shape = (frame_length, MFCC_NB_COEFFICIENTS, 1)

    # data files
    h5s = [
        timit_h5_fname('test', DataMode.mfcc, noise=True),
        # timit_h5_fname('train', DataMode.mfcc, noise=True),
        # vctk_h5_fname(DataMode.mfcc_delta)
        fff_en_h5_fname(DataMode.mfcc, noise=True),
        ffh_jp_h5_fname(DataMode.mfcc, noise=True),
    ]
    h5s = [os.path.join('./h5s', f) for f in h5s]
    name_h5_tuples = []
    for name in h5s:
        if name.find('train') == -1:
            name_h5_tuples.append((name, h5py.File(name, 'r', driver='core')))
        else:
            name_h5_tuples.append((name, h5py.File(name, 'r')))

    return input_shape, name_h5_tuples


def train3_softmax(name,
                   frame_length=None,
                   nb_epoch=1000,
                   batch_size=1,
                   early_stopping_patience=42,
                   chkd_dir=None):
    if chkd_dir is None:
        chkd_dir = CHECKPT_DIR

    input_shape, h5s = _train3_setup(frame_length)

    data_generators, sample_sizes, output_size = \
        get_speaker_generators_softmax(
            h5s, frame_length, batch_size)

    model = build_model6_softmax(
        input_shape, output_size, checkpoints_dir=chkd_dir)

    loss = 'categorical_crossentropy'
    all_metrics = ['accuracy']

    with open('./%s/model_%s.json' % (CHECKPT_DIR, name), 'w') as f:
        f.write(model.to_json())

    optim = RMSprop(lr=0.005, decay=1e-6)  # make_optimizer(**adam())
    model.compile(optimizer=optim, loss=loss, metrics=all_metrics)

    callbacks = [
        # ReduceLROnPlateau(
        # patience=early_stopping_patience / 2,
        # cooldown=early_stopping_patience / 4,
        # verbose=1),
        EarlyStopping(patience=early_stopping_patience, verbose=1),
    ]

    history_csv = 'train3_softmax_history_%s.csv' % time.strftime('%d_%m_%Y')
    callbacks.extend([
        ModelCheckpoint(
            os.path.join(CHECKPT_DIR,
                         'checkpoint.{epoch:05d}.va{val_acc:.5f}.hdf5'),
            save_best_only=True,
            monitor='val_acc'),
        CSVLogger(os.path.join('.', history_csv))
    ])

    glog.info('Generator param:: %s',
              pformat(
                  dict(
                      epochs=nb_epoch,
                      sample_sizes=sample_sizes,
                      batch_size=batch_size)))

    model.fit_generator(
        data_generators['train'],
        sample_sizes['train'] / batch_size,
        epochs=nb_epoch,
        validation_data=data_generators['test'],
        validation_steps=sample_sizes['test'] / batch_size,
        workers=4,
        callbacks=callbacks,
        initial_epoch=model.epoch_num)


def train3_e2e(name,
               frame_length,
               nb_epoch=1000,
               batch_size=72,
               early_stopping_patience=24,
               checkpoints_dir=None):
    input_shape, h5s = _train3_setup(frame_length)


if __name__ == '__main__':
    import fire

    if os.environ.get('DEBUG'):
        glog.setLevel('DEBUG')

    fire.Fire({
        't': train,
        't2': train2,
        't3': train3_softmax,
        # 'build_speaker_h5': build_speaker_h5,
        # 'build_embedding': build_embedding,
    })
