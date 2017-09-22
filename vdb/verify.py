#!/usr/bin/env python
import hashlib
import json
import os
from collections import defaultdict
from itertools import product, groupby
from pprint import pformat

import glog
import h5py
import librosa
import numpy as np
from keras import topology
from keras.models import model_from_json, Model
from python_speech_features import mfcc, logfbank, delta, ssc

from constant import (
    VCTK_ROOT,
    DAN_ROOT,
    TIMIT_ROOT,
    NOISE_ROOT,
    FFF_EN_ROOT,
    FFH_JP_ROOT,
    VOICE_ROOT,
    CN_ROOT, )
from data_util import MFCC_CONFIG
from generator import pad_data
from model import _last_checkpoint_path

_CACHE_DIR = './_verify_cache'
_spk_id = {}
_rev_look_up = {}
_counter = 0


def _wav_ids(wavs):
    return [hashlib.md5(w).hexdigest() for w in wavs]


def _get_spk_id(spk):
    global _counter, _spk_id

    if spk not in _spk_id:
        _spk_id[spk] = _counter
        _rev_look_up[_counter] = spk
        _counter += 1

    return _spk_id['spk']


def _decode_spk_id(spk_id):
    return _rev_look_up[spk_id]


def _wav_data(items, sr=8000):
    processed = []
    for d in items:
        data, sr = librosa.core.load(d, sr=sr)
        data, _ = librosa.effects.trim(data, top_db=15)
        data = mfcc(data, sr, **MFCC_CONFIG)
        processed.append(data)

    return processed


def _load_model(model_dir):
    model = None
    h5_path = _last_checkpoint_path(model_dir)

    with open(os.path.join(model_dir, 'model.json')) as m:
        model = model_from_json(m.read())

    model = Model(
        inputs=model.get_layer(name='start_utter'),
        outputs=model.get_layer(name='dvector'))

    model.load_weights(h5_path, by_name=True)
    return model


def _faiss(dim=128):
    index = faiss.IndexFlatIP(dim)
    index2 = faiss.IndexIDMap(index)
    return index, index2


def _verify(model_dir, verify_config_json, k=3):
    model = _load_model(model_dir)

    with open(verify_config_json) as f:
        verify_config = json.load(f)

    cache_id = hashlib.md5(model_dir + verify_config_json).hexdigest()
    cache_h5 = h5py.File(os.path.join(_CACHE_DIR, cache_id + '.h5'), 'w')

    def _flow(items):
        ids = []
        data = []
        for spk, wavs in items:
            spk_id = _get_spk_id(spk)
            wav_ids = _wav_ids(wavs)

            vecs = []
            for idx, w_id in enumerate(wav_ids):
                wav = wavs[idx]
                if ('/%s' % w_id) not in cache_h5:
                    wav = _wav_data(wav)
                    padded = pad_data(wav)
                    vec = model.predict_batch(np.array(padded, dtype='f32'))[0]
                    assert vec.shape == (128, )
                else:
                    vec = cache_h5['/%s' % w_id].value()
                vecs.append(vec)

            ids.append(spk_id)
            data.extend(vecs)

        return np.array(ids, dtype='int64'), np.array(data, dtype='f32')

    _, vec_db = _faiss()

    enroll_ids, enroll_data = _flow(verify_config['enroll'])
    vec_db.add_with_ids(enroll_data, enroll_ids)

    utter_ids, utter_data = _flow(verify_config['utter'])

    correct = 0
    gender = 0
    for idx, utter_id in enumerate(utter_ids):
        utter_spk = _decode_spk_id(utter_id)
        utter_vec = utter_data[idx]
        D, I = vec_db.search([utter_vec], k)
        decoded_I = [_decode_spk_id(i) for i in I]

        if utter_spk in decoded_I:
            correct += 1

        glog.info('D: %s', D)
        glog.info('I: %s', decoded_I)

    glog.info('Accuracy:: %s', correct / len(utter_ids))


if __name__ == '__main__':
    import fire

    fire.Fire({
        'm': _load_model,
    })
