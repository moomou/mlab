#!/usr/bin/env python
import math
import numpy as np
import os
from keras.utils import to_categorical

from util import (
    DATA_DIR,
    WS_IND,
    DEFAULT_CHAR_IND,
    LANGS as langs,
    NEWLINE_SYMBOL,
    allow_only_char,
    doc_to_word_index,
    get_raw_data_filename,
    get_data_filename,
    sample_k_line_from_file,
    sample_k_snippet_from_line,
)


def build_dataset(count, snippet_mode=False, k=100):
    x = open(get_data_filename('x', k, snippet_mode), 'w')
    y = open(get_data_filename('y', k, snippet_mode), 'w')

    buffers = []
    ys = []

    for index, lang in enumerate(langs):
        for i in range(count):
            with open(os.path.join(DATA_DIR, get_raw_data_filename(lang, i))) as f:
                sample_lines = sample_k_line_from_file(
                    f,
                    sample_count=k,
                    snippet_mode=snippet_mode
                )
                buffers.extend(sample_lines)
                ys.extend([str(index)] * len(sample_lines))

    cleaned = [allow_only_char(line) for line in buffers]

    x.write('\n'.join(cleaned))
    y.write('\n'.join(ys))

    print('Generated %d samples' % len(cleaned))


def build_test_dataset(snippet_mode=False):
    x = open(get_data_filename('x_test', 1, snippet_mode), 'w')
    y = open(get_data_filename('y_test', 1, snippet_mode), 'w')

    buffers = []
    ys = []

    for index, lang in enumerate(langs):
        with open(os.path.join(DATA_DIR, get_raw_data_filename(lang, 10))) as f:
            sample_lines = sample_k_line_from_file(
                f,
                sample_count=80,
                snippet_mode=snippet_mode
            )
            buffers.extend(sample_lines)
            ys.extend([str(index)] * len(sample_lines))

    cleaned = [allow_only_char(line) for line in buffers]

    x.write('\n'.join(cleaned))
    y.write('\n'.join(ys))

    print('Generated %d samples' % len(cleaned))


def load_dataset(test_mode=False, snippet_mode=False, k=2000, max_len=None):
    x_prefix = 'x' + ('_test' if test_mode else '')
    y_prefix = 'y' + ('_test' if test_mode else '')

    x_fname = get_data_filename(x_prefix, k, snippet_mode)
    y_fname = get_data_filename(y_prefix, k, snippet_mode)

    with open(y_fname) as f:
        labels = [int(i) for i in f]

    y = np.array(labels).reshape((len(labels), 1))
    X = np.ones((len(y), max_len), dtype=np.int64) * WS_IND

    with open(x_fname) as docs:
        for i, doc in enumerate(docs):
            X[i] = doc_to_word_index(doc, max_len=max_len)

    # shuffle
    ids = np.arange(len(X))
    np.random.shuffle(ids)

    X = X[ids]
    y = y[ids]

    print('Sample X::', X[:10])
    print('Sample Y::', y[:10])

    # save 15% as validation dataset
    train_len = math.floor(len(X) * 0.85)

    X_train = X[:train_len]
    y_train = to_categorical(y[:train_len], len(langs))

    X_test = X[train_len:]
    y_test = to_categorical(y[train_len:], len(langs))

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


def convert_to_k_snippet(X, Y, k=1):
    X_snippet = []
    Y_snippet = []

    for idx, x in enumerate(X):
        sampled = sample_k_snippet_from_line(x, k)
        X_snippet.extend(sampled)
        Y_snippet.extend([Y[idx]] * k)

    return (np.array(X_snippet), np.array(Y_snippet))


if __name__ == '__main__':
    build_dataset(10, k=256, snippet_mode=True)
    # build_dataset(10, k=2000)
    # build_test_dataset(snippet_mode=True)
    # build_test_dataset()
