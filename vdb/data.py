#!/usr/bin/env python
import os

import h5py
from tqdm import tqdm, trange

from data_util import process_wav
from constant import VCTK_ROOT, TIMIT_ROOT, SAMPLE_RATE


def _process_speaker(speaker, wavfiles, h5, mfcc=False):
    counter = 0
    for fname in tqdm(wavfiles):
        data = process_wav(fname, use_ulaw=not mfcc, mfcc=mfcc)
        h5.create_dataset(speaker + '.%s' % counter, data=data, dtype='int16')
        counter += 1

    h5.create_dataset(speaker + '.total', data=[counter], dtype='int32')


def timit(dataset, mfcc=False):
    fname = 'timit2_%s' % dataset
    if mfcc:
        fname += '_mfcc_@%s.h5' % SAMPLE_RATE
    else:
        fname += '_@%s.h5' % SAMPLE_RATE

    with h5py.File(fname, mode='a') as h5:
        for i in trange(1, 9):
            root = os.path.join(TIMIT_ROOT, dataset, 'DR%s' % i)
            root, speakers = next(os.walk(root))[:2]
            for speaker in tqdm(speakers):
                wavfiles = [
                    os.path.join(root, speaker, f)
                    for f in os.listdir(os.path.join(root, speaker))
                    if f.endswith('WAV')
                ]
                _process_speaker(speaker, wavfiles, h5, mfcc)


def vctk(mfcc=False):
    fname = 'vctk2.h5' if not mfcc else 'vctk2_mfcc.h5'
    with h5py.File(fname, mode='a') as h5:
        root, speakers = next(os.walk(VCTK_ROOT))[:2]
        for speaker in tqdm(speakers):
            wavfiles = [
                os.path.join(root, speaker, f)
                for f in os.listdir(os.path.join(root, speaker))
                if f.endswith('wav')
            ]
            _process_speaker(speaker, wavfiles, h5, mfcc)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'vctk': vctk,
        'timit': timit,
    })
