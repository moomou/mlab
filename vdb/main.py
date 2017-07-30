#!/usr/bin/env python
import os

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
    vctk_h5_fname, )
from model import (
    extend_model,
    build_model,
    build_model2,
    build_model3,
    load_model_h5,
    compute_receptive_field, )
from generator import one_hot
from pipeline import get_speaker_generators, get_wav_generators


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


def skip_out_of_receptive_field(func, width):
    def wrapper(y_true, y_pred):
        y_true = y_true[:, width - 1:, :]
        y_pred = y_pred[:, width - 1:, :]

        return func(y_true, y_pred)

    wrapper.__name__ = func.__name__
    return wrapper


def build_speaker_h5(output_path,
                     frame_stride=512,
                     frame_length=1535,
                     batch_count=5000,
                     batch_size=64,
                     one_hotify=False):

    train_h5 = h5py.File('./timit2_TRAIN.h5', 'r', driver='core')
    test_h5 = h5py.File('./timit2_TEST.h5', 'r', driver='core')

    data_generators, sample_sizes, output_size = get_speaker_generators(
        train_h5,
        test_h5,
        frame_length,
        frame_stride,
        batch_size,
        use_test_only=True,
        use_random=True,
        one_hotify=one_hotify)

    counter = 0
    generator = data_generators['test']
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('dimension', np.array([batch_size, frame_length, 1]))

        grp = f.create_group('speaker')
        while counter < batch_count:
            batch_inputs, batch_outputs = next(generator)
            # glog.info(batch_inputs.shape, batch_inputs[:10])
            speaker = batch_outputs.reshape(batch_size, 1)
            # glog.info(speaker.shape, speaker[:10])
            data = np.concatenate((batch_inputs, speaker), axis=-1)
            # glog.info(data.shape, data[:10])
            grp.create_dataset(str(counter), data=data, dtype='int32')
            counter += 1


def build_embedding(model_dir,
                    speaker_h5,
                    output_h5,
                    embedding_layer='embedding',
                    prediction_batch_size=256):
    model = load_model_h5(model_dir, {
        'categorical_mean_squared_error':
        categorical_mean_squared_error
    })

    embedding_model = keras.models.Model(
        inputs=model.input, outputs=model.get_layer(embedding_layer).output)

    with h5py.File(output_h5, 'w') as output:
        vector = output.create_group('vector')
        _id = output.create_group('_id')

        counter = 0
        with h5py.File(speaker_h5, 'r', driver='core') as f:
            dim = f['dimension'].shape
            grp = f['speaker']

            p_batch_frames = None
            p_speaker_id = None
            for key in tqdm(grp.keys()):
                data = grp[key]
                batch_frames = data[:, :dim[1]]
                speaker_id = data[:, dim[1]]

                if p_batch_frames is None:
                    p_batch_frames = batch_frames
                    p_speaker_id = speaker_id
                else:
                    p_batch_frames = np.concatenate((p_batch_frames,
                                                     batch_frames))
                    p_speaker_id = np.concatenate((p_speaker_id, speaker_id))

                if p_batch_frames.shape[0] < prediction_batch_size:
                    continue
                else:
                    assert p_batch_frames.shape[0] == p_speaker_id.shape[0]

                embedding_output = embedding_model.predict_on_batch(
                    one_hot(p_batch_frames))

                # the model provides a vector for each frame, join them up
                # embedding_shape = (batch_size, frame_length, 2048)
                embedding_output = np.average(embedding_output, axis=1)

                vector.create_dataset(
                    str(counter), data=embedding_output, dtype='float32')
                _id.create_dataset(
                    str(counter), data=p_speaker_id, dtype='int16')

                p_speaker_id = p_batch_frames = None
                counter += 1


def train(name,
          nb_epoch=1000,
          batch_size=12,
          kernel_size=3,
          nb_filter=64,
          dilation_depth=9,
          nb_stack=2,
          early_stopping_patience=20,
          train_only_in_receptive_field=False,
          test_only=False,
          debug=True,
          checkpoints_dir=None,
          target_mode='wav',
          input_mode='raw'):

    _set_tf_session()

    receptive_field_width, receptive_field_ms = \
            compute_receptive_field(dilation_depth, nb_stack)

    if input_mode == 'raw':
        glog.info('loading raw...')
        train_h5 = h5py.File('./timit2_TRAIN.h5', 'r', driver='core')
        test_h5 = h5py.File('./timit2_TEST.h5', 'r', driver='core')

        frame_stride = int(receptive_field_width / 2)
        frame_length = receptive_field_width * 4
    else:
        glog.info('loading mfcc...')
        train_h5 = h5py.File(
            './timit2_TRAIN_mfcc_@16000.h5', 'r', driver='core')
        test_h5 = h5py.File('./timit2_TEST_mfcc_@16000.h5', 'r', driver='core')

        mfcc_receptive_width = int(receptive_field_ms / MFCC_SAMPLE_LEN_MS)

        frame_stride = 15 * MFCC_NB_COEFFICIENTS
        frame_length = (30) * MFCC_NB_COEFFICIENTS

    generator = get_wav_generators \
                if target_mode == 'wav' \
                else get_speaker_generators

    data_generators, sample_sizes, output_size = generator(
        train_h5,
        test_h5,
        frame_length,
        frame_stride,
        batch_size,
        use_test_only=test_only)

    model = build_model(
        frame_length,
        nb_filter,
        dilation_depth,
        nb_stack,
        kernel_size=kernel_size,
        # hardcoding this to 256 for now
        nb_y_size=output_size,
        l2=0,
        checkpoints_dir=checkpoints_dir)

    if target_mode == 'speaker':
        loss = 'categorical_crossentropy'
        all_metrics = [
            'accuracy',
        ]
    else:
        loss = categorical_crossentropy
        all_metrics = [
            categorical_mean_squared_error,
            'accuracy',
        ]

    with open('./%s/model_%s.json' % (CHECKPT_DIR, name), 'w') as f:
        f.write(model.to_json())

    optim = make_optimizer(**adam())

    if train_only_in_receptive_field:
        loss = skip_out_of_receptive_field(loss, receptive_field_width)
        all_metrics = [
            skip_out_of_receptive_field(m, receptive_field_width)
            for m in all_metrics
        ]

    model.compile(optimizer=optim, loss=loss, metrics=all_metrics)

    callbacks = [
        ReduceLROnPlateau(
            patience=early_stopping_patience / 2,
            cooldown=early_stopping_patience / 4,
            verbose=1),
        EarlyStopping(patience=early_stopping_patience, verbose=1),
    ]

    callbacks.extend([
        ModelCheckpoint(
            os.path.join(CHECKPT_DIR,
                         'checkpoint.{epoch:05d}.va{val_acc:.5f}.hdf5'),
            save_best_only=True,
            monitor='val_acc'),
        CSVLogger(os.path.join('.', 'history.csv')),
    ])

    glog.info('???:: %s', sample_sizes)
    model.fit_generator(
        data_generators['train'],
        sample_sizes['train'] / batch_size,
        epochs=nb_epoch,
        validation_data=data_generators['test'],
        validation_steps=sample_sizes['test'] / batch_size,
        workers=4,
        callbacks=callbacks,
        initial_epoch=model.epoch_num)


def train2(name,
           nb_epoch=1000,
           batch_size=12,
           early_stopping_patience=20,
           train_only_in_receptive_field=False,
           test_only=False,
           debug=True,
           checkpoints_dir=None,
           mode='raw'):

    _set_tf_session()
    mode = getattr(DataMode, mode)

    glog.info('loading %s...' % mode.name)
    train_fname = timit_h5_fname('train', mode)
    test_fname = timit_h5_fname('test', mode)

    frame_length = 9
    frame_stride = 3

    train_h5 = h5py.File(train_fname, 'r', driver='core')
    test_h5 = h5py.File(test_fname, 'r', driver='core')

    generator = get_speaker_generators
    data_generators, sample_sizes, output_size = generator(
        train_h5,
        test_h5,
        frame_length,
        frame_stride,
        batch_size,
        use_test_only=test_only)

    model = build_model3(
        (frame_length, MFCC_NB_COEFFICIENTS),
        output_size,
        checkpoints_dir=checkpoints_dir)

    loss = 'categorical_crossentropy'
    all_metrics = [
        'accuracy',
    ]

    with open('./%s/model_%s.json' % (CHECKPT_DIR, name), 'w') as f:
        f.write(model.to_json())

    optim = make_optimizer(**adam())
    model.compile(optimizer=optim, loss=loss, metrics=all_metrics)

    callbacks = [
        ReduceLROnPlateau(
            patience=early_stopping_patience / 2,
            cooldown=early_stopping_patience / 4,
            verbose=1),
        EarlyStopping(patience=early_stopping_patience, verbose=1),
    ]

    callbacks.extend([
        ModelCheckpoint(
            os.path.join(CHECKPT_DIR,
                         'checkpoint.{epoch:05d}.va{val_acc:.5f}.hdf5'),
            save_best_only=True,
            monitor='val_acc'),
        CSVLogger(os.path.join('.', 'history.csv')),
    ])

    model.fit_generator(
        data_generators['train'],
        sample_sizes['train'] / batch_size,
        epochs=nb_epoch,
        validation_data=data_generators['test'],
        validation_steps=sample_sizes['test'] / batch_size,
        workers=4,
        callbacks=callbacks,
        initial_epoch=model.epoch_num)


if __name__ == '__main__':
    import fire
    import os

    if os.environ.get('DEBUG'):
        glog.setLevel('DEBUG')

    fire.Fire({
        't': train,
        't2': train2,
        'build_speaker_h5': build_speaker_h5,
        'build_embedding': build_embedding,
    })
