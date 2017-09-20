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
    elif diff < 0:
        # pick a random starting spot
        start = np.random.randint(0, abs(diff))
        assert start + frame_length < cur_frame_len
        return data[start:start + frame_length]

    assert diff > 0
    return np.pad(data, ((diff, 0), (0, 0), (0, 0)), 'symmetric')


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


def _random_speaker_wavfiles(h5_name, wav_names, enroll_k):
    spk_fnames = np.random.choice(wav_names, enroll_k + 1, replace=False)

    return [(h5_name, wav) for wav in spk_fnames]


def random_speaker_wavfiles(spk, files_by_speaker, enroll_k):
    h5_name, wav_names = files_by_speaker[spk]
    spk_wavfiles = _random_speaker_wavfiles(h5_name, wav_names, enroll_k)
    return spk_wavfiles


def _speaker_pair_generator(files_by_speaker, all_speakers, enroll_k,
                            batch_size):
    '''to ensure representative data sample
    we generate samples as follows:

    For every speaker, we keep positive sample and negative sample

        5 positive sample
        5 negative sample

    randomly
    '''
    all_positive_keys = []
    all_negative_keys = []

    for i in range(batch_size // 2):
        spk, utter_spk = np.random.choice(all_speakers, 2, replace=False)

        h5_name, wav_names = files_by_speaker[spk]
        spk_wavfiles = _random_speaker_wavfiles(h5_name, wav_names, enroll_k)
        all_positive_keys.append([(spk, [spk_wavfiles[0]]),
                                  (spk, spk_wavfiles[1:])])

        h5_name, wav_names = files_by_speaker[utter_spk]
        utter_spk_wavfiles = _random_speaker_wavfiles(h5_name, wav_names, 0)
        all_negative_keys.append([(utter_spk, [utter_spk_wavfiles[0]]),
                                  (spk, spk_wavfiles[1:])])

    return all_positive_keys, all_negative_keys


def _speaker_triplet_generator(files_by_speaker, all_speakers, enroll_k,
                               batch_size):
    '''to ensure representative data sample
    we generate samples as follows:

    For every speaker, we keep positive sample and negative sample

    all_keys = [
        (x, xp, xn),
    ]
    '''
    all_keys = []

    for i in range(batch_size // 2):
        spk, other_spk = np.random.choice(all_speakers, 2, replace=False)

        spk_wavfiles = random_speaker_wavfiles(spk, files_by_speaker, enroll_k)
        other_spk_wavfiles = random_speaker_wavfiles(
            other_spk, files_by_speaker, enroll_k)

        all_keys.append((spk_wavfiles[0], spk_wavfiles[1],
                         other_spk_wavfiles[0]))

    return all_keys


@threadsafe_generator
def speaker_batch_generator_e2e(h5_by_fname,
                                enroll_k,
                                files_by_speaker,
                                speaker_keys,
                                frame_length,
                                total_speaker,
                                batch_size,
                                norm=True):

    while True:
        all_positive_keys, all_negative_keys = _speaker_pair_generator(
            files_by_speaker, speaker_keys, enroll_k, batch_size)
        all_keys = all_positive_keys + all_negative_keys

        np.random.shuffle(all_keys)

        batch_inputs = {
            'start_utter': [],
        }
        for i in range(enroll_k):
            batch_inputs['start_enroll_%d' % i] = []
        batch_output_vec = []
        batch_output_bin = []

        for (speaker, spk_fname_data_tuple), \
                (enroll_speaker, enroll_spk_fname_data_tuple) \
                in all_keys:
            glog.debug('THERE: %s', enroll_spk_fname_data_tuple)
            enroll_data = [
                h5_by_fname[h5_fname][data_key].value
                for (h5_fname, data_key) in enroll_spk_fname_data_tuple
            ]
            glog.debug('HERE: %s', spk_fname_data_tuple)
            utter_data = [
                h5_by_fname[h5_fname][data_key].value
                for (h5_fname, data_key) in spk_fname_data_tuple
            ]

            if frame_length is not None:
                enroll_data = [
                    pad_data(data, frame_length) for data in enroll_data
                ]
                utter_data = [
                    pad_data(data, frame_length) for data in utter_data
                ]

            if norm:
                enroll_data = [
                    data / np.linalg.norm(data, axis=1)[..., np.newaxis]
                    for data in enroll_data
                ]
                utter_data = [
                    data / np.linalg.norm(data, axis=1)[..., np.newaxis]
                    for data in utter_data
                ]

            glog.debug('data:: %s', enroll_data[0].shape, len(enroll_data))
            glog.debug('data:: %s', utter_data[0].shape, len(utter_data))

            assert len(utter_data) == 1

            batch_inputs['start_utter'].append(utter_data[0])
            for i in range(enroll_k):
                batch_inputs['start_enroll_%d' % i].append(enroll_data[i])

            out = 1 if speaker == enroll_speaker else 0
            batch_output_vec.append(out)
            batch_output_bin.append(out)

        x = {k: np.array(v) for k, v in batch_inputs.items()}

        batch_output_vec = np.array(batch_output_vec)
        batch_output_bin = np.array(batch_output_bin)
        y = {
            'bin_out': batch_output_bin,
            'vec_out': batch_output_vec,
            'cosdist': batch_output_vec
        }

        yield x, y


@threadsafe_generator
def speaker_batch_generator_e2e_triplet(h5_by_fname,
                                        enroll_k,
                                        files_by_speaker,
                                        speaker_keys,
                                        frame_length,
                                        total_speaker,
                                        batch_size,
                                        norm=True):
    def _fetch_data(data_tuple):
        data = [
            h5_by_fname[h5_fname][data_key].value
            for (h5_fname, data_key) in data_tuple
        ]

        if frame_length is not None:
            data = [pad_data(d, frame_length) for d in data]
        if norm:
            data = [
                data / np.linalg.norm(data, axis=1)[..., np.newaxis]
                for d in data
            ]

        return data

    while True:
        all_keys = _speaker_triplet_generator(files_by_speaker, speaker_keys,
                                              enroll_k, batch_size)

        np.random.shuffle(all_keys)

        batch_inputs = {
            'x': [],
            'xp': [],
            'xn': [],
        }
        batch_output = []

        for x_fname_data_tuple, xp_fname_data_tuple, xn_fname_data_tuple in all_keys:
            glog.debug('THERE: %s', x_fname_data_tuple)
            x_data = _fetch_data(x_fname_data_tuple)
            xp_data = _fetch_data(xp_fname_data_tuple)
            xn_data = _fetch_data(x_fname_data_tuple)

            glog.debug('data:: %s', x_data[0].shape, len(x_data))

            batch_inputs['x'].append(x_data)
            batch_inputs['xp'].append(xp_data)
            batch_inputs['xn'].append(xn_data)

            batch_output.append((0, 0, 0))

        x = {k: np.array(v) for k, v in batch_inputs.items()}
        y = {'merged': batch_output}

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
