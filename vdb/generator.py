#!/usr/bin/env python
import threading

import numpy as np
import h5py
import glog


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def one_hot(x, count=256):
    return np.eye(count, dtype='uint8')[x.astype('uint8')]


def pad_data(data, frame_length):
    assert len(data.shape) == 3

    cur_frame_len = data.shape[0]
    diff = frame_length - cur_frame_len

    if diff == 0:
        return data

    assert diff > 0

    return np.pad(data, ((diff, 0), (0, 0), (0, 0)), 'constant')


@threadsafe_generator
def speaker_random_batch_generator(h5ref,
                                   nb_speaker,
                                   speaker_fn,
                                   keys,
                                   frame_length,
                                   batch_size,
                                   one_hotify=True):
    lengths = {key: h5ref[key].shape[0] for key in keys}

    while True:
        seq_indices = np.random.randint(0, len(keys), batch_size)
        batch_inputs = []
        batch_outputs = []

        for i, seq_i in enumerate(seq_indices):
            key = keys[seq_i]
            length = lengths[key]

            offset = np.squeeze(np.random.randint(0, length - frame_length, 1))
            batch_inputs.append(h5ref[key][offset:offset + frame_length])
            batch_outputs.append(speaker_fn(key))

        if one_hotify:
            yield one_hot(np.array(batch_inputs, dtype='uint8')), \
                one_hot(np.array(batch_outputs, dtype='uint8'), count=nb_speaker),
        else:
            yield (np.array(batch_inputs, dtype='uint8'), np.array(
                batch_outputs, dtype='uint8'), )


@threadsafe_generator
def wav_random_batch_generator(h5ref, keys, frame_length, batch_size):
    lengths = {key: h5ref[key].shape[0] for key in keys}

    while True:
        seq_indices = np.random.randint(0, len(keys), batch_size)
        batch_inputs = []
        batch_outputs = []

        for i, seq_i in enumerate(seq_indices):
            key = keys[seq_i]
            length = lengths[key]

            offset = np.squeeze(np.random.randint(0, length - frame_length, 1))
            batch_inputs.append(h5ref[key][offset:offset + frame_length])
            batch_outputs.append(
                h5ref[key][offset + 1:offset + frame_length + 1])

        yield one_hot(np.array(batch_inputs, dtype='uint8')), \
            one_hot(np.array(batch_outputs, dtype='uint8')),


def _generate_all_frame_by_key(keys,
                               lengths,
                               frame_length,
                               frame_stride,
                               per_key_limit=None,
                               shuffle=True):
    glog.debug('FrameLen:: %s' % frame_length)
    glog.debug('FrameStride:: %s' % frame_stride)
    key_frames = []

    for key in keys:
        length = lengths[key]
        for start in range(0, length - frame_length - 1, frame_stride):
            key_frames.append((key, start, start + frame_length))

    if shuffle:
        np.random.shuffle(key_frames)

    glog.debug('Key len:: %s' % len(keys))
    glog.debug('Key frame len:: %s' % len(key_frames))
    glog.debug('Key frame samples:: %s' % key_frames[:5])
    return key_frames


@threadsafe_generator
def wav_batch_generator(h5ref, keys, frame_length, frame_stride, batch_size):
    lengths = {key: h5ref[key].shape[0] for key in keys}
    key_frames = _generate_all_frame_by_key(keys, lengths, frame_length,
                                            frame_stride)

    counter = 0
    while True:
        batch_inputs = []
        batch_outputs = []

        for (key, start, end) in key_frames[counter:counter + batch_size]:
            batch_inputs.append(h5ref[key][start:end])
            batch_outputs.append(h5ref[key][start + 1:end + 1])

        counter += batch_size
        if counter + batch_size >= len(key_frames):
            np.random.shuffle(key_frames)
            counter = 0

        yield one_hot(np.array(batch_inputs, dtype='uint8')), \
            one_hot(np.array(batch_outputs, dtype='uint8')),


@threadsafe_generator
def speaker_batch_generator(h5ref,
                            nb_speaker,
                            speaker_fn,
                            keys,
                            frame_length,
                            frame_stride,
                            batch_size,
                            one_hotify=True,
                            mfcc=False):

    glog.debug('Total speaker:: %s' % nb_speaker)

    lengths = {key: h5ref[key].shape[0] for key in keys}
    key_frames = _generate_all_frame_by_key(keys, lengths, frame_length,
                                            frame_stride)

    counter = 0
    while True:
        batch_inputs = []
        batch_outputs = []

        for (key, start, end) in key_frames[counter:counter + batch_size]:
            batch_inputs.append(h5ref[key][start:end])
            batch_outputs.append(speaker_fn(key))

        counter += batch_size
        if counter + batch_size >= len(key_frames):
            np.random.shuffle(key_frames)
            counter = 0

        yield np.array(batch_inputs), \
            one_hot(np.array(batch_outputs, dtype='uint8'), count=nb_speaker),


@threadsafe_generator
def speaker_batch_generator_softmax(h5_by_fname,
                                    all_keys,
                                    frame_length,
                                    total_speaker,
                                    batch_size,
                                    norm=True):
    np.random.shuffle(all_keys)

    counter = 0
    while True:
        batch_inputs = []
        batch_outputs = []

        for speaker_id, h5_fname, data_key, in all_keys[counter:
                                                        counter + batch_size]:
            data = h5_by_fname[h5_fname][data_key].value

            if norm:
                data /= np.linalg.norm(data, axis=1)[..., np.newaxis]

            glog.debug('data:: %s', data.shape)
            if frame_length is not None:
                if data.shape[0] > frame_length:
                    data = data[:frame_length]
                elif data.shape[0] < frame_length:
                    data = pad_data(data, frame_length)

            batch_inputs.append(data)
            batch_outputs.append(speaker_id)

        counter += batch_size
        if counter + batch_size >= len(all_keys):
            np.random.shuffle(all_keys)
            counter = 0

        x = np.array(batch_inputs)
        y = one_hot(
            np.array(batch_outputs, dtype='uint16'), count=total_speaker)

        glog.debug('x:: %s', x.shape)
        glog.debug('y:: %s', y.shape)

        yield x, y


@threadsafe_generator
def speaker_batch_generator_e2e(
        h5_by_fname,
        all_positive_keys,  # spk = enroll spk
        all_negative_keys,  # spk != enroll spk
        frame_length,
        total_speaker,
        batch_size,
        norm=True):

    all_keys = all_positive_keys + all_negative_keys
    np.random.shuffle(all_keys)

    counter = 0
    while True:
        batch_inputs = []
        batch_outputs = []

        for same_speaker, \
                (speaker, spk_fname_data_tuple), \
                (enroll_speaker, enroll_spk_fname_data_tuple) \
                in all_keys[counter: counter + batch_size]:

            enroll_data = [
                h5_by_fname[h5_fname][data_key].value
                for (h5_fname, data_key) in enroll_spk_fname_data_tuple
            ]
            test_data = [
                h5_by_fname[h5_fname][data_key].value
                for (h5_fname, data_key) in enroll_spk_fname_data_tuple
            ]

            if frame_length is not None:
                enroll_data = [(pad_data(data, frame_length)
                                if len(data) < frame_length else data)
                               for data in enroll_data]
                test_data = [(pad_data(data, frame_length)
                              if len(data) < frame_length else data)
                             for data in test_data]

            if norm:
                enroll_data = [
                    data / np.linalg.norm(data, axis=1)[..., np.newaxis]
                    for data in enroll_data
                ]
                test_data = [
                    data / np.linalg.norm(data, axis=1)[..., np.newaxis]
                    for data in test_data
                ]

            glog.debug('data:: %s', enroll_data[0].shape, len(enroll_data))
            glog.debug('data:: %s', test_data[0].shape, len(test_data))

            assert len(test_data) == 1
            batch_inputs.append(np.array(test_data + enroll_data))
            # (True, False)
            batch_outputs.append(1 if same_speaker else 0)

        counter += batch_size
        if counter + batch_size >= len(all_keys):
            np.random.shuffle(all_keys)
            counter = 0

        x = batch_inputs
        y = one_hot(np.array(batch_outputs, dtype='uint8'), count=2)

        glog.debug('x:: %s', x.shape)
        glog.debug('y:: %s', y.shape)

        yield x, y


def full_wav_data(h5ref, keys, frame_length, frame_stride, batch_size):
    size = wav_batch_sample_sizes(h5ref, keys, frame_length, frame_stride,
                                  batch_size)
    gen = wav_random_batch_generator(h5ref, keys, frame_length, frame_stride,
                                     batch_size)

    all_data = []
    while len(all_data) < len(size):
        all_data.append(next(gen))

    return all_data


def wav_batch_sample_sizes(h5ref, keys, frame_length, frame_stride,
                           batch_size):
    lengths = {key: h5ref[key].shape[0] for key in keys}
    key_frames = _generate_all_frame_by_key(keys, lengths, frame_length,
                                            frame_stride)
    return len(key_frames)


def _test_wav_generate():
    with h5py.File('timit_TEST.h5') as h5:
        keys = [k for k in h5.keys() if not k.endswith('total')]
        gen = wav_random_batch_generator(h5, keys, 1024, 16, 256)
        sample = next(gen)
        glog.info(sample[0].shape, sample[1].shape)


if __name__ == '__main__':
    import fire
    fire.Fire({'test_wav_gen': _test_wav_generate})
