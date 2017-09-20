import keras.backend as K
import tensorflow as tf


def categorical_mean_squared_error(y_true, y_pred):
    '''MSE for categorical variables.'''
    return K.mean(
        K.square(K.argmax(y_true, axis=-1) - K.argmax(y_pred, axis=-1)))


def binary_accuracy(dist=0.99):
    def _binary_accuracy(y_true, y_pred):
        true = y_true == 1
        pred = y_pred >= dist

        return K.mean(K.equal(true, pred))

    return _binary_accuracy


def d_hinge_loss(p=1., n=1., dist=0.99, margin=0.3):
    # bin_loss_fn = binary_accuracy(dist)

    def _d_hinge_loss(y_true, y_pred):
        # scale dot product from [-1, 1] to [0, 1]
        scaled_y_pred = 0.5 * (y_pred + 1.0)

        match_loss = p * (dist - scaled_y_pred)
        mismatch_loss = K.maximum(0., margin - scaled_y_pred)

        vec_loss = K.mean(
            y_true * match_loss + (1. - y_true) * mismatch_loss, axis=-1)

        # bin_loss = bin_loss_fn(y_true, y_pred)

        return vec_loss

    return _d_hinge_loss


def d_logit_loss(p=1., n=1., dist=0.99, margin=0.3):
    def _d_logit_loss(y_true, y_pred):
        # convert from similarity to distance (from dot product to cosine distance)
        pred_dist = 1 - 0.5 * (y_pred + 1.0)

        px = 4. * (K.exp(pred_dist)) / K.square(1. + K.exp(pred_dist))

        return tf.losses.log_loss(y_true, px)

    return _d_logit_loss


def d_triplet_hinge_loss(d, dist=0.95, margin=0.3):
    def _d_triplet_hinge_loss(y_true, y_pred):
        xs = K.l2_normalize(y_pred[:, :d], -1)
        xps = K.l2_normalize(y_pred[:, d:(d + d)], -1)
        xns = K.l2_normalize(y_pred[:, -d:], -1)

        x_xp = K.batch_product(xs, xps)
        scaled_x_xp = 0.5 * (x_xp + 1.0)

        x_xn = K.batch_product(xs, xns)
        scaled_x_xn = 0.5 * (x_xn + 1.0)

        match_loss = dist - scaled_x_xp
        mismatch_loss = K.maximum(0., margin - scaled_x_xn)

        vec_loss = K.mean(match_loss + mismatch_loss)

        return vec_loss

    return _d_triplet_hinge_loss


def d_triplet_softmax_loss(d):
    def _d_triplet_softmax_loss(y_true, y_pred):
        # y_pred: [(x, xp, xn), (x, xp, xn), ...]
        # convert from similarity to distance (from dot product to cosine distance)
        xs = K.l2_normalize(y_pred[:, :d], -1)
        xps = K.l2_normalize(y_pred[:, d:(d + d)], -1)
        xns = K.l2_normalize(y_pred[:, -d:], -1)

        x_xp = K.batch_product(xs, xps)
        x_xn = K.batch_product(xs, xns)

        exp_x_xp = K.exp(x_xp)
        exp_x_xn = K.exp(x_xn)

        d_plus = exp_x_xp / (exp_x_xp + exp_x_xn)

        return K.square(d_plus)

    return _d_triplet_softmax_loss
