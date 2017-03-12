#!/usr/bin/env python
import glob
import os
import sys
import random

import h5py
import numpy as np

# ORDER CANNOT CHANGE
langs = [
    't_javascript.h5',
    't_python.h5',
    't_cpp.h5',
    't_php.h5',
    't_c.h5',
    't_csharp.h5',
    't_css.h5',
    't_sql.h5',
    't_java.h5',
    't_golang.h5',
    't_scala.h5',
    't_ruby.h5',
    't_swift.h5',
    't_html.h5',
    't_shell.h5',
]

SEQ_LEN = 50
def open_files(mode='r'):
    test_X = h5py.File('./test_X.72.h5', mode)
    test_y = h5py.File('./test_y.72.h5', mode)

    train_y = h5py.File('./train_y.72.h5', mode)
    train_X = h5py.File('./train_X.72.h5', mode)

    return (test_y, test_X, train_y, train_X)

def get_test_train(folder):
    t_y, t_x, tr_y, tr_x = open_files(mode='w')

    test_X = np.zeros((15000, SEQ_LEN, 50))
    test_Y = np.zeros((15000, 1))
    train_X = np.zeros((150000, SEQ_LEN, 50))
    train_Y = np.zeros((150000, 1))

    test_idx = 0
    train_idx = 0
    for y, f in enumerate(langs):
        print(y, f)
        h = h5py.File(os.path.join(folder, f))
        ks = [k for k in h.keys()]

        random.shuffle(ks)
        shuffled_ks = ks[:400]

        file_max = 1000
        for i in range(0, 50):
            doc_id = shuffled_ks[i]
            doc_value = h[doc_id].value

            max_len = doc_value.shape[0]
            assert doc_value.shape[1] == 50

            for j in range(100):
                start = random.randint(0, max_len)
                end = start + SEQ_LEN

                if end >= max_len: continue
                if test_idx >= 15000: break

                test_X[test_idx] = doc_value[start:end]
                test_Y[test_idx] = [y]

                test_idx += 1
                file_max -= 1

                if file_max == 0: break

            if file_max == 0: break

        file_max = 10000
        for i in range(51, 400):
            if i >= len(shuffled_ks): break

            doc_id = shuffled_ks[i]
            doc_value = h[doc_id].value

            max_len = doc_value.shape[0]

            for j in range(100):
                start = random.randint(0, max_len)
                end = start + SEQ_LEN

                if end >= max_len: continue
                if train_idx >= 150000: break

                train_X[train_idx] = doc_value[start:end]
                train_Y[train_idx] = [y]

                train_idx += 1
                file_max -= 1

                if file_max == 0: break

            if file_max == 0: break

        print('test_idx', test_idx)
        print('train_idx', train_idx)

    t_y.create_dataset('/data', data=test_Y, dtype=test_Y.dtype)
    t_x.create_dataset('/data', data=test_X, dtype=test_X.dtype)

    tr_y.create_dataset('/data', data=train_Y, dtype=train_Y.dtype)
    tr_x.create_dataset('/data', data=train_X, dtype=train_X.dtype)

    t_y.close()
    t_x.close()

    tr_y.close()
    tr_x.close()

def fix_Ys(filename):
    with h5py.File(filename, 'w') as y:
        print([k for k in y.keys()])
        print(y['/data'])
        print(y == 1)

if '__main__' == __name__:
    folder = sys.argv[1]

    get_test_train(folder)
