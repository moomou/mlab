#!/usr/bin/env python
import os
from enum import Enum

import scipy.io.wavfile as wavfile
import librosa
import numpy as np
import glog
from python_speech_features import mfcc, fbank, logfbank

from constant import VCTK_ROOT, SAMPLE_RATE, MFCC_NB_COEFFICIENTS

MFCC_CONFIG = {
    'numcep': MFCC_NB_COEFFICIENTS,
}

FBANK_CONFIG = {
    'nfilt': 40,
}

DATA_VERSION = 3


class DataMode(Enum):
    mfcc = 1
    fbank = 2
    raw = 3


def timit_h5_fname(dataset, mode):
    fname = 'timit%d_%s_%s_@%s.h5' % (DATA_VERSION, dataset.lower(), mode.name,
                                      SAMPLE_RATE, )
    return fname


def vctk_h5_fname(mode):
    fname = 'vctk%d_%s_%s_@%s.h5' % (DATA_VERSION, mode.name, SAMPLE_RATE, )
    return fname


def ulaw(x, u=255):
    '''Compress to ulaw; output e= [-1.0, 1.0]; x e= [-1.0, 1.0]'''
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


def _wav_to_float(x):
    '''Converts to [-1.0, 1.0]'''
    max_value = np.iinfo(x.dtype).max
    min_value = np.iinfo(x.dtype).min

    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= (max_value - min_value)

    x *= 2
    x -= 1

    return x


def float_to_uint8(x):
    # Rescale range to [0, 1]
    x += 1.
    x /= 2.

    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def process_wav(filename, sr=SAMPLE_RATE, mode=DataMode.raw):
    data, sr = librosa.core.load(filename, sr=sr)

    if mode == DataMode.mfcc:
        data = mfcc(data, sr, **MFCC_CONFIG)
        glog.debug('mfcc:: %s, %s', data.shape, data)
    elif mode == DataMode.fbank:
        data = logfbank(data, sr, **FBANK_CONFIG)
        glog.debug('fbank:: %s, %s', data.shape, data)
    else:
        data = ulaw(data)
        data = float_to_uint8(data)
        glog.debug('raw with ulaw:: %s, %s', data.shape, data)

    return data


def _ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio


def _ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = librosa.core.resample(mono_audio, file_sample_rate,
                                           desired_sample_rate)
    return mono_audio


def _process_wav(filename):
    channels = wavfile.read(filename)
    sr, audio = channels

    audio = _ensure_mono(audio)
    print('1::', audio[:100])
    audio = _wav_to_float(audio)
    print('2::', audio[:100])
    audio = ulaw(audio)
    print('3::', audio[:100])
    audio = _ensure_sample_rate(SAMPLE_RATE, sr, audio)
    print('4::', audio[:100])
    audio = float_to_uint8(audio)
    print('5::', audio[:100])


if __name__ == '__main__':
    import fire
    sample_file = os.path.join(VCTK_ROOT, 'p225', 'p225_001.wav')
    fire.Fire({
        't2':
        lambda: _process_wav(sample_file),
        't':
        lambda: process_wav(sample_file, mode=DataMode.mfcc).all() and 'fin',
    })
