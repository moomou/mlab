#!/usr/bin/env python
import sys
from collections import defaultdict
from itertools import groupby

import numpy as np
import glog

from generator import (
    wav_batch_sample_sizes,
    wav_batch_generator,
    speaker_random_batch_generator,
    speaker_batch_generator,
    speaker_batch_generator_softmax,
    speaker_batch_generator_e2e,
    _speaker_pair_generator, )
from speaker_id import create_speaker_id, get_total_speaker


class H5Wrapper:
    def __init__(self, *h5s):
        self.h5s = h5s

    def __getitem__(self, k):
        for h5 in self.h5s:
            if k in h5:
                return h5[k]


def _make_speaker(keys):
    speakers = set([key.split('.')[0] for key in keys])
    speakers = sorted(list(speakers))
    speaker_map = {speaker: num for num, speaker in enumerate(speakers)}

    def speaker_fn(key):
        spk = key.split('.')[0]
        return speaker_map[spk]

    return speaker_fn, speaker_map


def get_speaker_generators_softmax(all_h5s, frame_length, batch_size):
    all_test_keys = []
    all_train_keys = []
    h5_by_fname = {}

    for fname, h5 in all_h5s:
        h5_by_fname[fname] = h5
        all_keys = sorted([k for k in h5.keys() if not k.endswith('total')])
        glog.info('Here:: %s, %s', fname, len(all_keys))

        for k, g in groupby(all_keys, lambda k: k.split('.')[0]):
            # TODO: fix common prefix
            speaker_id = create_speaker_id(fname[:13], k)

            glist = list(g)
            # cannot shuffl - if you did this, then validation
            # and training data get mixed up
            np.random.shuffle(glist)

            train_len = int(len(glist) * 0.8)
            test_len = len(glist) - train_len

            assert test_len > 0 and train_len > 0

            # files_by_speaker_train[speaker_id] = (fname, glist[:train_len])
            all_train_keys.extend(
                zip([speaker_id] * train_len, [fname] * train_len,
                    glist[:train_len]))

            # files_by_speaker_test[speaker_id] = (fname, glist[train_len:])
            all_test_keys.extend(
                zip([speaker_id] * test_len, [fname] * test_len, glist[
                    train_len:]))

    total_speaker = get_total_speaker()
    glog.debug('Total speaker:: %s' % total_speaker)

    return {
        'train':
        speaker_batch_generator_softmax(
            h5_by_fname,
            all_train_keys,
            frame_length,
            total_speaker,
            batch_size=batch_size),
        'test':
        speaker_batch_generator_softmax(
            h5_by_fname,
            all_test_keys,
            frame_length,
            total_speaker,
            batch_size=batch_size),
    }, {
        'train': len(all_train_keys),
        'test': len(all_test_keys),
    }, total_speaker


def get_speaker_generators_e2e(all_h5s,
                               frame_length,
                               batch_size,
                               enroll_k,
                               train_size=50e3):
    all_speaker_files = defaultdict(list)
    train_speakers = None
    test_speakers = None
    h5_by_fname = {}

    for fname, h5 in all_h5s:
        h5_by_fname[fname] = h5
        all_keys = sorted([k for k in h5.keys() if not k.endswith('total')])
        glog.info('Here:: %s, %s', fname, len(all_keys))

        for k, g in groupby(all_keys, lambda k: k.split('.')[0]):
            # TODO: fix common prefix
            speaker_id = create_speaker_id(fname[:13], k)

            glist = list(g)
            assert len(glist) > 0

            if len(glist) >= enroll_k + 1:
                all_speaker_files[speaker_id] = (fname, glist)

    all_keys = list(all_speaker_files.keys())
    train_len = int(len(all_keys) * 0.8)
    train_speakers = all_keys[:train_len]
    test_speakers = all_keys[train_len:]

    total_speaker = get_total_speaker()
    glog.debug('Total speaker:: %s' % total_speaker)

    return {
        'train':
        speaker_batch_generator_e2e(
            h5_by_fname,
            enroll_k,
            all_speaker_files,
            train_speakers,
            frame_length,
            total_speaker,
            batch_size=batch_size),
        'test':
        speaker_batch_generator_e2e(
            h5_by_fname,
            enroll_k,
            all_speaker_files,
            test_speakers,
            frame_length,
            total_speaker,
            batch_size=batch_size),
    }, {
        'train': int(train_size),
        'test': int(train_size / 5),
    }, total_speaker


def get_speaker_generators(train_h5,
                           test_h5,
                           frame_length,
                           frame_stride,
                           batch_size,
                           one_hotify=True,
                           use_test_only=False,
                           use_random=False):
    train_keys = [k for k in train_h5.keys() if not k.endswith('total')]
    test_keys = [k for k in test_h5.keys() if not k.endswith('total')]

    if use_test_only:
        glog.info('Using test only...')
        data_keys = test_keys
        test_h5 = train_h5
    else:
        glog.info('Combining test and train...')
        data_keys = train_keys + test_keys
        combined_h5ref = H5Wrapper(train_h5, test_h5)
        train_h5 = combined_h5ref
        test_h5 = combined_h5ref

    speaker_fn, speaker_map = _make_speaker(data_keys)
    keys_by_speaker = defaultdict(list)
    for k in data_keys:
        speaker = speaker_fn(k)
        keys_by_speaker[speaker].append(k)

    train_keys = []
    test_keys = []

    # TODO: remove this
    counter = 0
    for speaker in keys_by_speaker:
        sample_len = len(keys_by_speaker[speaker])
        np.random.shuffle(keys_by_speaker[speaker])

        if sample_len < 2:
            glog.error('Skipping speaker found::', speaker)
            sys.exit(1)

        test_len = int(0.1 * sample_len)
        test_keys.extend(keys_by_speaker[speaker][:test_len])
        train_keys.extend(keys_by_speaker[speaker][test_len:])
        counter += 1

    np.random.shuffle(train_keys)
    np.random.shuffle(test_keys)

    glog.info('Total train keys:: %s', len(train_keys))
    glog.debug('Sample train keys:: %s', train_keys[:10])
    glog.info('Total test keys:: %s', len(test_keys))
    glog.debug('Sample test keys:: %s', test_keys[:10])

    generator = speaker_random_batch_generator \
        if use_random \
        else speaker_batch_generator

    return {
        'train':
        generator(
            train_h5,
            len(speaker_map),
            speaker_fn,
            train_keys,
            frame_length,
            frame_stride,
            batch_size=batch_size,
            one_hotify=one_hotify),
        'test':
        generator(
            test_h5,
            len(speaker_map),
            speaker_fn,
            test_keys,
            frame_length,
            frame_stride,
            batch_size=batch_size,
            one_hotify=one_hotify),
    }, {
        'train':
        wav_batch_sample_sizes(train_h5, train_keys, frame_length,
                               frame_stride, batch_size),
        'test':
        wav_batch_sample_sizes(test_h5, test_keys, frame_length, frame_stride,
                               batch_size),
    }, len(speaker_map)


def get_wav_generators(train_h5,
                       test_h5,
                       frame_length,
                       frame_stride,
                       batch_size,
                       use_test_only=False):
    test_keys = [k for k in test_h5.keys() if not k.endswith('total')]

    if use_test_only:
        test_keys = test_keys[:int(0.7 * len(test_keys))]
        split = int(len(test_keys) * 0.12)
        train_keys = test_keys[split:]
        test_keys = test_keys[:split]
        train_h5 = test_h5
        glog.info('Train keys::', len(train_keys), train_keys[:10])
        glog.info('Test keys::', len(test_keys), test_keys[:10])
    else:
        train_keys = [k for k in train_h5.keys() if not k.endswith('total')]

    return {
        'train':
        wav_batch_generator(train_h5, train_keys, frame_length, frame_stride,
                            batch_size),
        'test':
        wav_batch_generator(test_h5, test_keys, frame_length, frame_stride,
                            batch_size),
    }, {
        'train':
        wav_batch_sample_sizes(train_h5, train_keys, frame_length,
                               frame_stride, batch_size),
        'test':
        wav_batch_sample_sizes(test_h5, test_keys, frame_length, frame_stride,
                               batch_size),
    }, 256


def _test_generator():
    import h5py
    from data_util import (DataMode, timit_h5_fname)

    train_fname = timit_h5_fname('train', DataMode.mfcc)
    test_fname = timit_h5_fname('train', DataMode.mfcc)

    train_h5 = h5py.File(train_fname, 'r', driver='core')
    test_h5 = h5py.File(test_fname, 'r', driver='core')

    data_generators, sample_sizes, output_size = get_speaker_generators(
        train_h5, test_h5, frame_length=9, frame_stride=3, batch_size=12)

    test_gen = data_generators['test']

    for (x, y) in test_gen:
        glog.info('TestGen x:: %s, %s' % (x.shape, x[:10]))
        glog.info('TestGen y:: %s' % np.argmax(y, axis=-1))

        break

    glog.info('Total Train Sizes:: %s' % sample_sizes['train'])
    glog.info('Total Speaker:: %s' % output_size)


if __name__ == '__main__':
    import fire
    import os
    if os.environ.get('DEBUG'):
        glog.setLevel('DEBUG')
    fire.Fire({'t': _test_generator})
