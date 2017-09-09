#!/usr/bin/env python

import h5py
import numpy as np

def check_h5(fname):
  seter = set()
  with h5py.File(fname) as f:
    for k in f.keys():
        spk, _ = k.split('.')
        seter.add(spk)
        assert not np.isnan(f[k].value).any(), 'Bad value:: %s' % k

    print(len(seter)

if __name__ == '__main__':
    import fire
    fire.Fire({
        'check': check_h5
    })
