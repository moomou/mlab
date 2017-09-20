#!/usr/bin/env python
import os

import glog
import h5py
import keras
from keras.models import (
    Model,
    load_model, )
from keras.layers.merge import (
    Dot as l_dot,
    add as l_add,
    multiply as l_multiply,
    average as l_average,
    concatenate as l_concat, )
from keras.layers import Dense, Dropout, Activation, LSTM, GRU, Reshape
from keras.layers.noise import GaussianNoise
from keras.layers import Embedding, Flatten, BatchNormalization
from keras import regularizers as reg
from keras.layers import (
    Conv1D,
    Conv2D,
    AveragePooling2D,
    MaxPooling1D,
    MaxPooling2D,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    LocallyConnected2D,
    Input, )

from constant import SAMPLE_RATE
from block import WavnetBlock, Fire1D, AvgLayer


def _last_checkpoint_path(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints = [f for f in checkpoints if f.endswith('hdf5')]

    if len(checkpoints) == 0:
        return None, None

    checkpoints.sort(
        key=lambda x: os.stat(os.path.join(checkpoints_dir, x)).st_mtime)
    last_checkpoint = checkpoints[-1]
    last_checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint)

    return last_checkpoint, last_checkpoint_path


def _load_checkoint(model, checkpoints_dir):
    last_checkpoint, last_checkpoint_path = _last_checkpoint_path(
        checkpoints_dir)

    if last_checkpoint and last_checkpoint_path:
        model.epoch_num = int(last_checkpoint[11:16]) + 1
        model.load_weights(last_checkpoint_path, by_name=True)

        glog.info('Loaded model from epoch: %d' % model.epoch_num)

        return True

    return None


def compute_receptive_field(dilation_depth, nb_stack, sr=SAMPLE_RATE):
    receptive_field = nb_stack * (2**dilation_depth * 2) - (nb_stack - 1)
    receptive_field_ms = (receptive_field * 1000) / sr
    return receptive_field, receptive_field_ms


def load_model_weight(model, model_dir):
    _load_checkoint(model, model_dir)
    return model


def load_model_h5(model_dir, custom_objects):
    last_checkpoint, last_checkpoint_path = _last_checkpoint_path(model_dir)
    return load_model(last_checkpoint_path, custom_objects=custom_objects)


def build_model(frame_length,
                nb_filter,
                dilation_depth,
                nb_stack,
                nb_output_bin=256,
                nb_y_size=256,
                use_bias=True,
                kernel_size=128,
                l2=0.01,
                checkpoints_dir=None):

    start_tensor = Input(shape=(frame_length, nb_output_bin), name='start')
    out = start_tensor
    skip_conn = []

    out = Conv1D(
        nb_filter, kernel_size, padding='causal',
        name='initial_causal_conv')(out)

    for s in range(nb_stack):
        for i in range(0, dilation_depth + 1):
            out, skip_out = WavnetBlock(
                nb_filter, kernel_size, s, i, l2=l2)(out)
            skip_conn.append(skip_out)

    out = l_add(skip_conn)
    out = Activation('relu')(out)
    out = Conv1D(
        3,
        3,
        padding='same',
        activation='relu',
        activity_regularizer=reg.l2(l2))(out)

    out = Conv1D(3, 3, padding='same', activity_regularizer=reg.l2(l2))(out)
    out = GlobalAveragePooling1D()(out)
    out = Dense(2**10, name='embedding')(out)
    out = Dense(nb_y_size, activation='softmax')(out)

    model = Model(inputs=start_tensor, outputs=out)

    field_width, field_ms = compute_receptive_field(dilation_depth, nb_stack)
    glog.info('model with width=%s and width_ms=%s' % (field_width, field_ms))
    model.summary()

    if checkpoints_dir:
        if not _load_checkoint(model, checkpoints_dir):
            model.epoch_num = 0
    else:
        model.epoch_num = 0

    return model


def build_model2(frame_length,
                 nb_filter,
                 dilation_depth,
                 nb_stack,
                 nb_output_bin=256,
                 nb_y_size=256,
                 use_bias=True,
                 kernel_size=128,
                 l2=0.01,
                 checkpoints_dir=None):

    start_tensor = Input(shape=(frame_length, nb_output_bin), name='start')
    out = start_tensor

    out = Conv1D(256, 1, padding='causal', name='start_conv')(out)

    for s in range(nb_stack):
        skip_conn = []

        out, _ = wavnet_res_block(256 // (s + 1), 3, s, s + 1)(out)
        for i in range(3):
            s11 = 2**(6 - s)
            e11 = e33 = 2**(9 - s)

            if i == 2:
                pass
                # out = l_add([out, skip_conn.pop()])

            out = fire_1d_block(s11, e11, e33,
                                'fire%s.%d.%d.%d' % (i, s11, e11, e33))(out)

        out = MaxPooling1D(pool_size=1, padding='valid')(out)

    out = Dropout(0.5)(out)
    out = GlobalMaxPooling1D()(out)
    out = Dense(nb_y_size, activation='softmax')(out)

    model = Model(inputs=start_tensor, outputs=out)

    field_width, field_ms = compute_receptive_field(dilation_depth, nb_stack)
    glog.info('model with width=%s and width_ms=%s' % (field_width, field_ms))

    model.summary()

    if checkpoints_dir:
        _load_checkoint(model, checkpoints_dir)
    else:
        model.epoch_num = 0

    return model


def build_model3(input_shape,
                 nb_output_bin,
                 kernel_sizes,
                 checkpoints_dir=None):
    # ref: Deep Speaker Feature Learning for Text-independent Speaker Verification
    shape = list(input_shape)
    glog.info('Shape:: %s', shape)
    start = Input(shape=shape, name='start')
    out = Conv2D(128, kernel_sizes.pop(0))(start)
    # shape = (6, 24) or (6, 33)
    out = MaxPooling2D(kernel_sizes.pop(0))(out)
    # shape = (3, 12)
    out = Conv2D(256, kernel_sizes.pop(0))(out)
    # shape = (2, 8)
    out = MaxPooling2D(kernel_sizes.pop(0))(out)
    # shape = (2, 4)
    out = Reshape(kernel_sizes.pop(0))(out)

    # bottleneck
    out = Dense(512, name='bottleneck')(out)

    dilation_depth = 3
    for i in range(dilation_depth):
        out = Conv1D(
            256,
            1,
            dilation_rate=2**i,
            name='relu_dilation_%s' % (2**i),
            padding='same',
            activation='relu')(out)

    # D vector
    out = Dense(400, name='d_vector')(out)
    out = GlobalAveragePooling1D()(out)
    out = Dense(nb_output_bin, activation='softmax')(out)

    model = Model(inputs=start, outputs=out)
    field_width, field_ms = compute_receptive_field(dilation_depth, 1)
    glog.info('model with width=%s and width_ms=%s' % (field_width, field_ms))
    model.summary()

    if checkpoints_dir:
        _load_checkoint(model, checkpoints_dir)
    else:
        model.epoch_num = 0

    return model


def build_model4(input_shape,
                 nb_output_bin,
                 kernel_sizes,
                 checkpoints_dir=None):
    # ref: Deep Speaker Feature Learning for Text-independent Speaker Verification

    shape = list(input_shape)
    dilation_depth = 4

    glog.info('Shape:: %s', shape)
    out = start = Input(shape=shape, name='start')

    delay_kernel_size = kernel_sizes.pop(0)
    for i in range(dilation_depth):
        out = Conv2D(
            128,
            delay_kernel_size,
            dilation_rate=(2**i, 1),
            name='relu_dilation_%s' % (2**i),
            padding='same',
            activation='relu')(out)

    out = Dropout(0.5)(out)
    out = Dense(512, name='bottleneck')(out)

    out = Conv2D(128, kernel_sizes.pop(0))(out)
    # shape = (6, 24) or (6, 33)
    out = MaxPooling2D(kernel_sizes.pop(0))(out)
    # shape = (3, 12)
    out = Conv2D(256, kernel_sizes.pop(0))(out)
    # shape = (2, 8)
    out = MaxPooling2D(kernel_sizes.pop(0))(out)

    print(out.shape)
    # shape = (3, 5)
    out = Reshape(kernel_sizes.pop(0))(out)
    out = Dropout(0.2)(out)

    # D vector
    out = Dense(400, name='d_vector')(out)
    out = GlobalAveragePooling1D()(out)
    out = Dense(nb_output_bin, activation='softmax')(out)

    model = Model(inputs=start, outputs=out)
    field_width, field_ms = compute_receptive_field(dilation_depth, 1)
    glog.info('model with width=%s and width_ms=%s' % (field_width, field_ms))
    model.summary()

    if checkpoints_dir:
        _load_checkoint(model, checkpoints_dir)
    else:
        model.epoch_num = 0

    return model


def build_model5(input_shape,
                 nb_output_bin,
                 kernel_sizes,
                 checkpoints_dir=None):
    # ref: Deep Speaker Feature Learning for Text-independent Speaker Verification
    shape = list(input_shape)
    dilation_depth = 4
    stack = 1

    glog.info('Shape:: %s', shape)
    out = start = Input(shape=shape, name='start')

    out = GaussianNoise(0.025)(out)

    delay_kernel_size = kernel_sizes.pop(0)
    for j in range(stack):
        for i in range(dilation_depth):
            out = Conv2D(
                128,
                delay_kernel_size,
                dilation_rate=(2**i, 1),
                name='relu_dilation_%s_%s' % (j, 2**i),
                padding='same',
                activation='relu')(out)

    out = Dense(512, name='bottleneck')(out)

    out = Conv2D(256, kernel_sizes.pop(0))(out)
    # shape = (6, 24) or (6, 33)
    out = MaxPooling2D(kernel_sizes.pop(0))(out)
    # shape = (3, 12)
    out = Conv2D(256, kernel_sizes.pop(0))(out)
    # shape = (2, 8)
    out = MaxPooling2D(kernel_sizes.pop(0))(out)

    # out = Conv2D(128, kernel_sizes.pop(0))(out)
    # out = Dropout(0.5)(out)
    # out = Conv2D(128, kernel_sizes.pop(0))(out)
    # out = Dropout(0.5)(out)

    print(out.shape)
    # shape = (3, 5)
    out = Reshape(kernel_sizes.pop(0))(out)

    # D vector
    out = Dense(400, name='d_vector')(out)
    out = GlobalAveragePooling1D()(out)
    out = Dense(nb_output_bin, activation='softmax')(out)

    model = Model(inputs=start, outputs=out)
    field_width, field_ms = compute_receptive_field(dilation_depth, 1)
    glog.info('model with width=%s and width_ms=%s' % (field_width, field_ms))
    model.summary()

    if checkpoints_dir:
        _load_checkoint(model, checkpoints_dir)
    else:
        model.epoch_num = 0

    return model


def build_model6_softmax(input_shape, nb_output_bin, checkpoints_dir=None):
    shape = list(input_shape)

    glog.info('Shape:: %s', shape)
    out = start = Input(shape=shape, name='start')
    out = GaussianNoise(0.025)(out)

    # TODO: try dilated convolution
    # TODO: try attention network
    out = Conv2D(2**6, (1, 3), activation='relu')(out)
    out = Conv2D(2**6, (2, 2), padding='same', activation='relu')(out)
    out = Dense(2**7, activation='relu')(out)
    out = Dense(2**7, activation='relu')(out)
    out = Dropout(0.5)(out)

    out = Dense(2**7, activation='relu')(out)
    out = Dropout(0.5)(out)
    # TODO: try residual network
    # TODO: try stacked bottle neck layer
    out = AveragePooling2D((1, 2**7), data_format='channels_first')(out)
    # out = AvgLayer()(out)
    out = Flatten()(out)
    out = Dense(2**7, name='dvec', activation='relu')(out)
    out = Dense(
        nb_output_bin, activation='softmax',
        name='output_%s' % nb_output_bin)(out)

    model = Model(inputs=start, outputs=out)
    model.summary()

    if checkpoints_dir:
        if not _load_checkoint(model, checkpoints_dir):
            model.epoch_num = 0
    else:
        model.epoch_num = 0

    return model


def build_model6_e2e(input_shape, enroll_k, checkpoints_dir=None):
    shape = list(input_shape)

    glog.info('Shape:: %s', shape)
    utter_out = utter_start = Input(shape=shape, name='start_utter')
    enroll_inputs = [
        Input(shape=shape, name='start_enroll_%d' % d) for d in range(enroll_k)
    ]
    enroll_outs = enroll_inputs

    # siamese architecture
    siamese_lower = [
        # 2** 7
        Conv2D(
            2**7, (3, 3),
            dilation_rate=(3, 1),
            padding='same',
            activation='relu'),
        Conv2D(
            2**7, (1, 1),
            dilation_rate=(3, 1),
            padding='same',
            activation='relu'),
    ]
    siamese_upper = [
        Dense(2**7, activation='relu'),
        Dropout(0.5),
        Dense(2**7, activation='relu'),
        Dropout(0.5),
        AveragePooling2D((1, 2**7), data_format='channels_first'),
        Flatten(),
        Dense(2**7, name='dvector', activity_regularizer=reg.l2(0.01)),
    ]

    for l in siamese_lower:
        utter_out = l(utter_out)
        enroll_outs = [l(eo) for eo in enroll_outs]

    # skip connection
    # utter_out = l_add([utter_out, utter_start])
    # enroll_outs = [
    # l_add([enroll_outs[idx], enroll_inputs[idx]])
    # for idx in range(enroll_k)
    # ]

    for l in siamese_upper:
        utter_out = l(utter_out)
        enroll_outs = [l(eo) for eo in enroll_outs]

    # average all enroll layers
    enroll_out = l_average(enroll_outs, name='enroll_output')

    cosdist = l_dot(
        axes=-1, normalize=True, name='cosdist')([enroll_out, utter_out])
    # bin_out = Dense(1, activation='sigmoid', name='bin_out')(cosdist)

    model = Model(
        inputs=([utter_start] + enroll_inputs),
        outputs=[
            # vec_out,
            # bin_out,
            cosdist,
        ])
    model.summary()

    if checkpoints_dir:
        if not _load_checkoint(model, checkpoints_dir):
            model.epoch_num = 0
    else:
        model.epoch_num = 0

    return model


def build_model7_e2e(input_shape, checkpoints_dir=None):
    shape = list(input_shape)

    glog.info('Shape:: %s', shape)

    x_out = x = Input(shape=shape, name='x')
    xp_out = xp = Input(shape=shape, name='xp')
    xn_out = xn = Input(shape=shape, name='xn')

    # siamese architecture
    siamese_lower = [
        Conv2D(
            2**7, (3, 3),
            dilation_rate=(3, 1),
            padding='same',
            activation='relu'),
        Conv2D(
            2**7, (1, 1),
            dilation_rate=(3, 1),
            padding='same',
            activation='relu'),
    ]
    siamese_upper = [
        Dense(2**7, activation='relu'),
        Dropout(0.5),
        Dense(2**7, activation='relu'),
        Dropout(0.5),
        AveragePooling2D((1, 2**7), data_format='channels_first'),
        Flatten(),
        Dense(2**7, name='dvector', activity_regularizer=reg.l2(0.01)),
    ]

    for l in siamese_lower:
        x_out = l(x_out)
        xp_out = l(xp_out)
        xn_out = l(xn_out)

    # skip connection
    # utter_out = l_add([utter_out, utter_start])
    # enroll_outs = [
    # l_add([enroll_outs[idx], enroll_inputs[idx]])
    # for idx in range(enroll_k)
    # ]

    for l in siamese_upper:
        x_out = l(x_out)
        xp_out = l(xp_out)
        xn_out = l(xn_out)

    merged = l_concat([x_out, xp_out, xn_out], axis=-1)

    model = Model(
        inputs=[x, xp, xn], outputs=[
            merged,
        ])
    model.summary()

    if checkpoints_dir:
        if not _load_checkoint(model, checkpoints_dir):
            model.epoch_num = 0
    else:
        model.epoch_num = 0

    return model


if __name__ == '__main__':
    import fire

    frame_length = 20
    frame_stride = 4
    input_shape = (frame_length, 52, 1)

    kernel_sizes = [
        (3, 3),  # dilation kernel x n - output: (frame_length, 52),
        (3, 3),  # conv2d - output: (18, 50)
        (2, 2),  # maxpool - output: (9, 25),
        (3, 3),  # conv2d - output: (7, 23)
        (2, 2),  # maxpool - output: (3, 11)
        (33, 128),  # reshape
    ]

    fire.Fire({
        'm': lambda: build_model5(input_shape, 100, kernel_sizes),
        '6e': lambda: build_model6_e2e((300, 40, 1), enroll_k=5),
        'crf': compute_receptive_field,
    })
