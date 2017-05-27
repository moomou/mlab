#!/usr/bin/env python
import math
import numpy as np
import os
from keras.utils import to_categorical

from util import (
    DATA_DIR,
    DEFAULT_CHAR_IND,
    LANGS as langs,
    NEWLINE_SYMBOL,
    allow_only_char,
    get_raw_data_filename,
    get_data_filename,
    remove_hash_comment,
    sample_k_line_from_file,
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


def load_dataset(max_len, snippet_mode=False, k=1000):
    x_fname = get_data_filename('x', k, snippet_mode)
    y_fname = get_data_filename('y', k, snippet_mode)

    with open(y_fname) as f:
        labels = [int(i) for i in f]

    y = np.array(labels).reshape((len(labels), 1))
    X = np.ones((len(y), max_len), dtype=np.int64) * -1

    with open(x_fname) as docs:
        for i, doc in enumerate(docs):
            for j, char in enumerate(doc[-max_len:]):
                char = char.lower()
                if char == '\n':
                    <char > < / char >= NEWLINE_SYMBOL
                char_ind = DEFAULT_CHAR_IND.get(char, -1)
                X[i, j] = char_ind

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

    print('x_test_shape::', X_test.shape)
    print('y_test_shape::', y_test.shape)

    print('x_train_shape::', X_train.shape)
    print('y_train_shape::', y_train.shape)

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


if __name__ == '__main__':
    # build_dataset(10, k=2000, snippet_mode=True)
    build_dataset(10, k=2000)
