#!/usr/bin/env python
import os
import multiprocessing as mp
from enum import Enum
from itertools import zip_longest

import glog
import librosa
import numpy as np
import pydub
from pydub import AudioSegment
from python_speech_features import mfcc, logfbank, delta, ssc

from cache import cache_run
from constant import VCTK_ROOT, SAMPLE_RATE, MFCC_NB_COEFFICIENTS
from config import THREAD_POOL

SSC_CONFIG = {}
MFCC_CONFIG = {
    'numcep': MFCC_NB_COEFFICIENTS,
}

FBANK_CONFIG = {
    'nfilt': 40,
}

S2N_RATIO = [5, 10]  # 10, 15
PLAY_BACK_SPEED = [1, 1.05]

DATA_VERSION = 6


class DataMode(Enum):
    summary = 0
    mfcc = 1
    fbank = 2
    raw = 3

    mfcc_delta = 4

    spec = 5
    mfcc_ssc = 6
    ssc = 7


def timit_h5_fname(dataset, mode, noise=True):
    fname = 'timit%d_%s_%s_@%s%s.h5' % (DATA_VERSION, dataset.lower(),
                                        mode.name, SAMPLE_RATE, ''
                                        if noise is None else '_n')
    return fname


def vctk_h5_fname(mode, noise=True):
    fname = 'vctk%d_%s_%s_@%s.h5' % (DATA_VERSION, mode.name, SAMPLE_RATE, ''
                                     if noise is None else '_n')
    return fname


def fff_en_h5_fname(mode, noise=True):
    return 'fff_en_%d_%s_@%s%s.h5' % (DATA_VERSION, mode.name, SAMPLE_RATE, ''
                                      if noise is None else '_n')


def ffh_jp_h5_fname(mode, noise=True):
    return 'ffh_jp_%d_%s_@%s%s.h5' % (DATA_VERSION, mode.name, SAMPLE_RATE, ''
                                      if noise is None else '_n')


def voice_h5_fname(mode, noise=True):
    return 'voice_%d_%s_@%s%s.h5' % (DATA_VERSION, mode.name, SAMPLE_RATE, ''
                                     if noise is None else '_n')


def split_file(filename):
    basename, ext = os.path.basename(filename).split('.')
    src = AudioSegment.from_file(filename)

    splits = []
    # len too short, simply skip
    if len(src) < 500:
        return []
    # if len > 8 sec, we split into segment of 2 sec
    if len(src) > 8 * 1000:
        # take the first 60 second for now
        src = src[:60 * 1000]
        # split into segment of 2 second
        segs = list(range(0, len(src), 2000))
        segs[-1] = len(src)

        for i in range(len(segs) - 1):

            def export(cache_path):
                src[segs[i]:segs[i + 1]].export(
                    cache_path,
                    format='wav',
                    bitrate='%dk' % (SAMPLE_RATE // 1000))

            splits.append(cache_run(basename + '_sp%s.%s' % (i, ext), export))
    else:
        splits.append(filename)

    return splits


def _sample_bg_noise(bg_noise, choice=1):
    keys = list(bg_noise.keys())
    choices = np.random.choice(len(keys), choice, replace=False)
    choices = [keys[c] for c in choices]

    return [(key, bg_noise[key]) for key in choices]


def augment_data(fnames, output_dir, aug_option):
    aug_option = aug_option or {}
    try:
        os.mkdir(output_dir)
    except:
        pass

    bg_noise = aug_option.get('bgNoise', {})
    bg_noise['silence'] = AudioSegment.silent()

    generated_fnames = []
    for filename in fnames:  # tqdm(fnames, desc='aug', position=3):
        fkey = os.path.basename(filename)
        seg = AudioSegment.from_file(filename)

        fname = 'fade_in_%s' % fkey
        path = os.path.join(output_dir, fname)
        generated_fnames.append(path)
        if not os.path.exists(path) or aug_option.get('overwrite'):
            seg.fade_in(duration=len(seg) // 4).export(
                path, format='wav', bitrate='%dk' % (SAMPLE_RATE // 1000))

        fname = 'fade_out_%s' % fkey
        path = os.path.join(output_dir, fname)
        generated_fnames.append(path)
        if not os.path.exists(path) or aug_option.get('overwrite'):
            seg.fade_out(duration=len(seg) // 4).export(
                path, format='wav', bitrate='%dk' % (SAMPLE_RATE // 1000))

        for speed in PLAY_BACK_SPEED:
            sped = pydub.effects.speedup(seg, playback_speed=speed)

            fname = 'sped%s_%s' % (speed, fkey)
            path = os.path.join(output_dir, fname)
            generated_fnames.append(path)

            if not os.path.exists(path) or aug_option.get('overwrite'):
                sped.export(
                    path, format='wav', bitrate='%dk' % (SAMPLE_RATE // 1000))

        # _sample_bg_noise(bg_noise, 3):
        # runs through all bg noise
        for noise_key, noise_segs in bg_noise.items():
            choice = np.random.randint(0, len(noise_segs))
            noise_seg = noise_segs[choice]

            for gain in S2N_RATIO:
                fname = '%s%s_g%s_%s' % (noise_key, choice, gain, fkey)
                path = os.path.join(output_dir, fname)
                generated_fnames.append(path)

                noised = seg.overlay(
                    noise_seg, loop=True, gain_during_overlay=gain)

                if not os.path.exists(path) or aug_option.get('overwrite'):
                    noised.export(
                        path,
                        format='wav',
                        bitrate='%dk' % (SAMPLE_RATE // 1000))

    assert len(generated_fnames) >= 1
    return generated_fnames


def _encode_data(args):
    filename, mode, sr = args

    data, sr = librosa.core.load(filename, sr=sr)
    duration = librosa.get_duration(y=data, sr=sr)

    if mode.name.startswith('spec'):
        data = np.log(abs(librosa.core.stft(y=data, n_fft=2**11))**2)
        data = data[..., np.newaxis]
        glog.debug('spec:: %s, %s', data.shape, data)
    elif mode.name.startswith('ssc'):
        data = ssc(data, sr, **SSC_CONFIG)
        data = data[..., np.newaxis]
        glog.debug('mfcc:: %s, %s', data.shape, data)
    elif mode.name.startswith('mfcc'):
        data = mfcc(data, sr, **MFCC_CONFIG)

        if mode == DataMode.mfcc_delta:
            data_delta = delta(data, 1)
            data = np.append(data, data_delta, axis=-1)
        elif mode == DataMode.mfcc_ssc:
            data = ssc(data, sr, **SSC_CONFIG)
            data = np.append(data, data_delta, axis=-1)

        data = data[..., np.newaxis]
        glog.debug('mfcc:: %s, %s', data.shape, data)
    elif mode == DataMode.fbank:
        data = logfbank(data, sr, **FBANK_CONFIG)
        data = data[..., np.newaxis]
        glog.debug('fbank:: %s, %s', data.shape, data)
    else:
        assert False, 'Invald option:: %s' % mode

    return data, duration


def process_wav(args):
    if len(args) == 3:
        sr = SAMPLE_RATE
        filename, mode, aug_option = args
    else:
        filename, mode, aug_option, sr = args

    mode = mode or DataMode.raw
    aug_option = aug_option or None

    fnames = split_file(filename)

    # files too short, skip
    if len(fnames) == 0:
        return 0, []

    aug_dir = os.path.join(os.path.dirname(filename), 'aug')

    if aug_option is not None and aug_option.get('bgNoise'):
        gen_fnames = augment_data(fnames, aug_dir, aug_option)
        fnames.extend(gen_fnames)

    all_data = []
    total_duration_sec = 0

    groups = zip_longest(*(iter(fnames), ) * mp.cpu_count())
    for group in groups:
        if THREAD_POOL != 'SPK':
            data_duration_tuple = p.map(_encode_data, [(g, mode, sr)
                                                       for g in group if g])
        else:
            data_duration_tuple = [
                _encode_data((g, mode, sr)) for g in group if g
            ]

        for data, duration in data_duration_tuple:
            all_data.append(data)
            total_duration_sec += duration

    return total_duration_sec, all_data


p = mp.Pool(mp.cpu_count() // 2)
if __name__ == '__main__':
    import fire

    if os.environ.get('DEBUG'):
        glog.setLevel('DEBUG')

    sample_file = os.path.join(VCTK_ROOT, 'p225', 'p225_001.wav')
    fire.Fire({
        't':
        lambda: process_wav(sample_file, mode=DataMode.mfcc_delta).all() and 'fin',
    })
