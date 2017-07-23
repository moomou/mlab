#!/usr/bin/env python
import threading

import numpy as np
import h5py


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
                               per_key_limit=None):
    key_frames = []

    for key in keys:
        length = lengths[key]
        for start in range(0, length - frame_length - 1, frame_stride):
            key_frames.append((key, start, start + frame_length))

    np.random.shuffle(key_frames)
    print('Key len::', len(keys))
    print('Key frame len::', len(key_frames))
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
                            one_hotify=True):

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

        yield one_hot(np.array(batch_inputs, dtype='uint8')), \
            one_hot(np.array(batch_outputs, dtype='uint8'), count=nb_speaker),


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


def test_wav_generate():
    with h5py.File('timit_TEST.h5') as h5:
        keys = [k for k in h5.keys() if not k.endswith('total')]
        gen = wav_random_batch_generator(h5, keys, 1024, 16, 256)
        sample = next(gen)
        print(sample[0].shape, sample[1].shape)


if __name__ == '__main__':
    import fire
    fire.Fire({'test_wav_gen': test_wav_generate})
