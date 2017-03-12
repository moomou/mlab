#!/usr/bin/env python

import sys
import time

from main import save_glove_h5py

if __name__ == '__main__':
    inputname = sys.argv[1]
    vec_dim = sys.argv[2]
    output = sys.argv[3]

    start = time.time()
    print('Saving to %s' % output)
    save_glove_h5py(inputname, vec_dim, output)
    print('Done:: %s', time.time() - start)
