from keras.layers.merge import (add as l_add, multiply as l_multiply)
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    LSTM,
    GRU,
    Concatenate, )
from keras.layers import Embedding, Flatten, BatchNormalization
from keras import regularizers as reg
from keras.layers import (
    Conv1D,
    GlobalMaxPooling1D,
    Input, )


def wavnet_res_block(nb_filter,
                     kernel_size,
                     stack_i,
                     dr_i,
                     l2=0.01,
                     padding='causal'):
    def f(input_tensor):
        dr = 2**dr_i

        residual = input_tensor

        tanh_out = Conv1D(
            nb_filter,
            kernel_size,
            padding=padding,
            dilation_rate=dr,
            name='dilate_tanh_d%d_s%s' % (dr, stack_i),
            activation='tanh',
            activity_regularizer=reg.l2(l2))(input_tensor)

        sigm_out = Conv1D(
            nb_filter,
            kernel_size,
            padding=padding,
            dilation_rate=dr,
            name='dilate_sigmoid_d%d_s%s' % (dr, stack_i),
            activation='sigmoid',
            activity_regularizer=reg.l2(l2))(input_tensor)

        merged = l_multiply([tanh_out, sigm_out])
        skip_out = Conv1D(
            nb_filter, 1, activation='relu', padding='same')(merged)

        out = l_add([skip_out, residual])
        return out, skip_out

    return f


def fire_1d_block(s11, e11, e33, name_prefix, dilation_rate=1, padding='same'):
    def l(input_layer):
        output = Conv1D(
            s11,
            1,
            activation='relu',
            kernel_initializer='glorot_uniform',
            padding=padding,
            dilation_rate=dilation_rate,
            name='%s_squeeze' % name_prefix)(input_layer)

        expand1 = Conv1D(
            e11,
            1,
            activation='relu',
            kernel_initializer='glorot_uniform',
            padding=padding,
            dilation_rate=dilation_rate,
            name='%s_expand1' % name_prefix)(output)

        expand2 = Conv1D(
            e33,
            3,
            activation='relu',
            kernel_initializer='glorot_uniform',
            padding=padding,
            dilation_rate=dilation_rate,
            name='%s_expand2' % name_prefix)(output)

        merged = l_add([expand1, expand2])
        # merged = Concatenate(
        # axis=1, name='%s_concat' % name_prefix)([expand1, expand2])

        return merged

    return l
