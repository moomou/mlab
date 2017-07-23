#!/usr/bin/env python
import os

from keras.models import (
    Model,
    load_model, )
from keras.layers.merge import (add as l_add, multiply as l_multiply)
from keras.layers import Dense, Dropout, Activation, LSTM, GRU
from keras.layers import Embedding, Flatten, BatchNormalization
from keras import regularizers as reg
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    GlobalMaxPooling1D,
    Input, )

from constant import SAMPLE_RATE
from block import wavnet_res_block, fire_1d_block


def _last_checkpoint_path(checkpoints_dir):
    checkpoints = os.listdir(checkpoints_dir)
    checkpoints.sort(
        key=lambda x: os.stat(os.path.join(checkpoints_dir, x)).st_mtime)
    last_checkpoint = checkpoints[-1]
    last_checkpoint_path = os.path.join(checkpoints_dir, last_checkpoint)

    return last_checkpoint, last_checkpoint_path


def _load_checkoint(model, checkpoints_dir):
    last_checkpoint, last_checkpoint_path = _last_checkpoint_path(
        checkpoints_dir)
    model.epoch_num = int(last_checkpoint[11:16]) + 1

    print('Loading model from epoch: %d' % model.epoch_num)
    model.load_weights(last_checkpoint_path)


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
            out, skip_out = wavnet_res_block(
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
    out = Dense(2048, name='embedding')(out)
    out = Dense(nb_y_size, activation='softmax')(out)

    model = Model(inputs=start_tensor, outputs=out)

    field_width, field_ms = compute_receptive_field(dilation_depth, nb_stack)
    print('model with width=%s and width_ms=%s' % (field_width, field_ms))

    model.summary()

    if checkpoints_dir:
        _load_checkoint(model, checkpoints_dir)
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
    print('model with width=%s and width_ms=%s' % (field_width, field_ms))

    model.summary()

    if checkpoints_dir:
        _load_checkoint(model, checkpoints_dir)
    else:
        model.epoch_num = 0

    return model


def extend_model(model, output_size):
    embedding_layer = model.layers[-1].output
    # out = GRU(64)(embedding_layer)
    out = GlobalMaxPooling1D()(embedding_layer)
    out = Dense(output_size, activation='softmax', name='output')(out)

    ext_model = Model(inputs=model.input, outputs=out)

    ext_model.epoch_num = 0
    ext_model.summary()
    return ext_model


if __name__ == '__main__':
    import fire
    fire.Fire({
        'model': lambda: build_model(1024, 2, 9, 3),
        'crf': compute_receptive_field,
    })
