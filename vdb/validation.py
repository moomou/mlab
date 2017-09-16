#!/usr/bin/env python
import json
import os
from collections import defaultdict
from itertools import product, groupby
from pprint import pformat

import glog
import librosa
import numpy as np

from constant import (
    VCTK_ROOT,
    DAN_ROOT,
    TIMIT_ROOT,
    NOISE_ROOT,
    FFF_EN_ROOT,
    FFH_JP_ROOT,
    VOICE_ROOT,
    CN_ROOT, )


def _index_list(array):
    return list(range(len(array)))


def _generate_utter_enroll_set(all_spk, enroll_k):
    utter_set = []
    enroll_set = []

    for spk in all_spk.keys():
        data = all_spk[spk]
        choices = np.random.choice(
            _index_list(data), 1 + enroll_k, replace=False)
        utter_set.append([spk, [data[choices[0]]]])
        enroll_set.append([spk, [data[i] for i in choices[1:]]])

    return utter_set, enroll_set


def _dan(enroll_k, dataset='part_1'):
    dataset_root = os.path.join(DAN_ROOT, dataset)
    root, groups, _ = next(os.walk(dataset_root))

    all_files = []
    for grp in groups:
        _, _, files = next(os.walk(os.path.join(root, grp)))
        all_files.extend([os.path.join(root, grp, f) for f in files])

    sorted(all_files)
    files_by_speaker = defaultdict(list)
    for key, group in groupby(all_files,
                              lambda k: os.path.basename(k).split('-')[0]):
        files_by_speaker[key].extend(group)

    return _generate_utter_enroll_set(files_by_speaker, enroll_k)


def _vctk(enroll_k):
    root, speakers = next(os.walk(VCTK_ROOT))[:2]

    files_by_speaker = {}
    for speaker in speakers:
        files_by_speaker[speaker] = [
            os.path.join(root, speaker, f)
            for f in os.listdir(os.path.join(root, speaker))
            if f.endswith('wav')
        ][:12]

    return _generate_utter_enroll_set(files_by_speaker, enroll_k)


def _cn(enroll_k):
    dataset_root = os.path.join(CN_ROOT, 'test')
    root, speakers, _ = next(os.walk(dataset_root))

    all_spk = {}
    for speaker in speakers:
        wavfiles = [
            os.path.join(root, speaker, f)
            for f in os.listdir(os.path.join(root, speaker))
            if f.lower().endswith('wav')
        ]

        # we take at most 10
        wavfiles = sorted(wavfiles)[:10]
        all_spk[speaker] = wavfiles

    return _generate_utter_enroll_set(all_spk, enroll_k)


def _voice(enroll_k):
    root, langs, _ = next(os.walk(VOICE_ROOT))

    all_files = []
    for lang in [lang for lang in langs if lang != 'cn']:
        _, _, files = next(os.walk(os.path.join(root, lang, 'data')))
        all_files.extend([os.path.join(root, lang, 'data', f) for f in files])

    sorted(all_files)
    files_by_speaker = defaultdict(list)
    for key, group in groupby(
            all_files,
            lambda k: (os.path.basename(k)[:11] + '_' + os.path.basename(k)[12:].split('_')[0])
    ):
        files_by_speaker[key].extend(group)

    segmented_files_by_speaker = defaultdict(list)
    for spk, files in files_by_speaker.items():
        for f in files:
            data, sr = librosa.core.load(f)
            duration = librosa.get_duration(y=data, sr=sr)

            # split into 5 file segments
            if duration > 5:
                for start_time in range(0, int(duration), 5):
                    segmented_files_by_speaker[spk].append((f, start_time,
                                                            start_time + 5))
            else:
                segmented_files_by_speaker[spk].append(f)

    return _generate_utter_enroll_set(segmented_files_by_speaker, enroll_k)


def gen(dataset, enroll_k=4):
    fn = {
        'cn': _cn,
        'voice': _voice,
        'vctk': _vctk,
        'dan': _dan,
    }[dataset]

    utter, enroll = fn(enroll_k)
    data = {
        'utter': utter,
        'enroll': enroll,
    }

    with open('validation_%s.yaml' % dataset, 'w') as f:
        f.write(json.dumps(data))


if __name__ == '__main__':
    import fire

    fire.Fire({'gen': gen})
