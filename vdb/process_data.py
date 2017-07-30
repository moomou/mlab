#!/usr/bin/env python
import os

import h5py
from tqdm import tqdm, trange

from data_util import DataMode, process_wav, timit_h5_fname, vctk_h5_fname
from constant import VCTK_ROOT, TIMIT_ROOT


def _process_speaker(speaker, wavfiles, h5, mode):
    counter = 0
    for fname in tqdm(wavfiles, desc='files'):
        data = process_wav(fname, mode=mode)
        h5.create_dataset(
            speaker + '.%s' % counter, data=data, dtype='float32')
        counter += 1

    h5.create_dataset(speaker + '.total', data=[counter], dtype='int32')


def timit(dataset, mode='raw'):
    mode = getattr(DataMode, mode)
    fname = timit_h5_fname(dataset, mode)

    with h5py.File(fname, mode='a') as h5:
        for i in trange(1, 9, desc='dir'):
            root = os.path.join(TIMIT_ROOT, dataset, 'DR%s' % i)
            root, speakers = next(os.walk(root))[:2]
            for speaker in tqdm(speakers):
                wavfiles = [
                    os.path.join(root, speaker, f)
                    for f in os.listdir(os.path.join(root, speaker))
                    if f.endswith('WAV')
                ]
                _process_speaker(speaker, wavfiles, h5, mode)


def vctk(mode='raw'):
    mode = getattr(DataMode, mode)
    fname = vctk_h5_fname(mode)

    with h5py.File(fname, mode='a') as h5:
        root, speakers = next(os.walk(VCTK_ROOT))[:2]
        for speaker in tqdm(speakers, desc='dir'):
            wavfiles = [
                os.path.join(root, speaker, f)
                for f in os.listdir(os.path.join(root, speaker))
                if f.endswith('wav')
            ]
            _process_speaker(speaker, wavfiles, h5, mode)


if __name__ == '__main__':
    import fire
    fire.Fire({
        'vctk': vctk,
        'timit': timit,
    })
