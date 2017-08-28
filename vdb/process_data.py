#!/usr/bin/env python
import os
import multiprocessing as mp
from itertools import groupby, zip_longest
from collections import defaultdict
from pprint import pformat

import h5py
import glog
from tqdm import tqdm, trange
from pydub import AudioSegment

from data_util import (
    DataMode,
    process_wav,
    timit_h5_fname,
    dan_h5_fname,
    vctk_h5_fname,
    ffh_jp_h5_fname,
    fff_en_h5_fname,
    voice_h5_fname,
    cn_wav_h5_fname, )

from constant import (
    VCTK_ROOT,
    DAN_ROOT,
    TIMIT_ROOT,
    NOISE_ROOT,
    FFF_EN_ROOT,
    FFH_JP_ROOT,
    VOICE_ROOT,
    CN_ROOT, )
from config import THREAD_POOL


def _process_speaker(speaker, wavfiles, h5, mode, aug_option=None):
    counter = 0
    total_duration_sec = 0

    wavfiles_group = zip_longest(*(iter(wavfiles), ) * mp.cpu_count())

    for names in tqdm(wavfiles_group, desc='files', position=1):
        if THREAD_POOL == 'SPK':
            results = p.map(process_wav, [(n, mode, aug_option) for n in names
                                          if n])
        else:
            results = [process_wav((n, mode, aug_option)) for n in names if n]

        for duration_sec, all_data in results:
            total_duration_sec += duration_sec

            for data in all_data:
                h5.create_dataset(
                    speaker + '.%s' % counter, data=data, dtype='float32')
                counter += 1

    h5.create_dataset(speaker + '.total', data=[counter], dtype='int32')
    return total_duration_sec


_all_bg_cache = {}


def _all_bg():
    global _all_bg_cache

    if len(_all_bg_cache) != 0:
        return _all_bg_cache

    for noise_type in tqdm(os.listdir(NOISE_ROOT), desc='load_noise'):
        noise_dir = os.path.join(NOISE_ROOT, noise_type)

        if not os.path.isdir(noise_dir):
            continue

        wav_paths = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir)]
        wav_paths = sorted(wav_paths)

        # TODO: limit to 3 for now
        wav_paths = wav_paths[:3]
        noise_key = os.path.basename(noise_type)

        _all_bg_cache[noise_key] = [
            AudioSegment.from_file(f) for f in wav_paths
        ]

    return _all_bg_cache


def timit(dataset, mode='raw', overwrite=True):
    mode = getattr(DataMode, mode)
    fname = timit_h5_fname(dataset, mode)

    all_bg = _all_bg()

    with h5py.File(fname, mode='a') as h5:
        speaker_stat = {}
        for i in trange(1, 9, desc='dir'):
            root = os.path.join(TIMIT_ROOT, dataset, 'DR%s' % i)
            root, speakers = next(os.walk(root))[:2]
            for speaker in tqdm(speakers):
                wavfiles = [
                    os.path.join(root, speaker, f)
                    for f in os.listdir(os.path.join(root, speaker))
                    if f.endswith('WAV')
                ]

                total_duration_sec = _process_speaker(
                    speaker,
                    wavfiles,
                    h5,
                    mode,
                    aug_option={
                        'bgNoise': all_bg,
                        'overwrite': overwrite,
                    })
                speaker_stat[speaker] = total_duration_sec

        glog.info('\n' * 10)
        glog.info(pformat(speaker_stat))


def vctk(mode='raw'):
    mode = getattr(DataMode, mode)
    fname = vctk_h5_fname(mode)

    all_bg = _all_bg()

    speaker_stat = {}
    with h5py.File(fname, mode='a') as h5:
        root, speakers = next(os.walk(VCTK_ROOT))[:2]

        for speaker in tqdm(speakers, desc='spk'):
            wavfiles = [
                os.path.join(root, speaker, f)
                for f in os.listdir(os.path.join(root, speaker))
                if f.endswith('wav')
            ][:12]

            duration_sec = _process_speaker(
                speaker, wavfiles, h5, mode, aug_option={
                    'bgNoise': all_bg,
                })

            speaker_stat[speaker] = duration_sec

    glog.info('\n' * 10)
    glog.info(pformat(speaker_stat))


def ffh_jp(mode='raw', overwrite=False):
    mode = getattr(DataMode, mode)
    h5_name = ffh_jp_h5_fname(mode)

    all_bg = _all_bg()
    with h5py.File(h5_name, mode='a') as h5:
        _, episodes = next(os.walk(FFH_JP_ROOT))[:2]

        all_files = []
        for episode in episodes:
            root, _, files = next(os.walk(os.path.join(FFH_JP_ROOT, episode)))
            all_files.extend([os.path.join(root, f) for f in files])

        sorted(all_files)
        files_by_speaker = defaultdict(list)
        for key, group in groupby(all_files,
                                  lambda k: os.path.basename(k).split('_')[1]):
            files_by_speaker[key].extend(group)

        speaker_stat = {}
        for speaker, wavfiles in tqdm(files_by_speaker.items(), desc='spk'):
            total_duration_sec = _process_speaker(
                speaker,
                wavfiles,
                h5,
                mode,
                aug_option={
                    'bgNoise': all_bg,
                    'overwrite': overwrite,
                })
            speaker_stat[speaker] = total_duration_sec

        glog.info('\n' * 10)
        glog.info(pformat(speaker_stat))


def fff_en(mode='raw', overwrite=True):
    mode = getattr(DataMode, mode)
    h5_name = fff_en_h5_fname(mode)

    speaker_stat = {}
    all_bg = _all_bg()

    with h5py.File(h5_name, mode='a') as h5:
        root, speakers = next(os.walk(FFF_EN_ROOT))[:2]

        for speaker in tqdm(speakers, desc='spk'):
            wavfiles = [
                os.path.join(root, speaker, f)
                for f in os.listdir(os.path.join(root, speaker))
                if f.lower().endswith('wav')
            ]
            total_duration_sec = _process_speaker(
                speaker,
                wavfiles,
                h5,
                mode,
                aug_option={
                    'bgNoise': all_bg,
                    'overwrite': overwrite,
                })
            speaker_stat[speaker] = total_duration_sec

        glog.info(pformat(speaker_stat))


def voice(mode='raw'):
    mode = getattr(DataMode, mode)
    h5_name = voice_h5_fname(mode)

    all_bg = _all_bg()
    with h5py.File(h5_name, mode='a') as h5:
        root, langs, _ = next(os.walk(VOICE_ROOT))

        all_files = []
        for lang in [lang for lang in langs if lang != 'cn']:
            _, _, files = next(os.walk(os.path.join(root, lang, 'data')))
            all_files.extend(
                [os.path.join(root, lang, 'data', f) for f in files])

        sorted(all_files)
        files_by_speaker = defaultdict(list)
        for key, group in groupby(
                all_files,
                lambda k: (os.path.basename(k)[:11] + '_' + os.path.basename(k)[12:].split('_')[0])
        ):
            files_by_speaker[key].extend(group)

        speaker_stat = {}
        for speaker, wavfiles in tqdm(files_by_speaker.items(), desc='spk'):
            total_duration_sec = _process_speaker(
                speaker, wavfiles, h5, mode, aug_option={
                    'bgNoise': all_bg,
                })
            speaker_stat[speaker] = total_duration_sec

        glog.info(pformat(speaker_stat))


def cn_wav(dataset, mode='raw', overwrite=True):
    mode = getattr(DataMode, mode)
    h5_name = cn_wav_h5_fname(dataset, mode)

    all_bg = _all_bg()
    dataset_root = os.path.join(CN_ROOT, dataset)
    with h5py.File(h5_name, mode='a') as h5:
        root, speakers, _ = next(os.walk(dataset_root))

        speaker_stat = {}
        for speaker in tqdm(speakers, desc='spk'):
            wavfiles = [
                os.path.join(root, speaker, f)
                for f in os.listdir(os.path.join(root, speaker))
                if f.lower().endswith('wav')
            ]

            # we take at most 10
            wavfiles = sorted(wavfiles)[:10]

            total_duration_sec = _process_speaker(
                speaker,
                wavfiles,
                h5,
                mode,
                aug_option={
                    'bgNoise': all_bg,
                    'overwrite': overwrite,
                })
            speaker_stat[speaker] = total_duration_sec

        glog.info('\n' * 10)
        glog.info(pformat(speaker_stat))


def danish_wav(dataset, mode='raw', overwrite=True):
    mode = getattr(DataMode, mode)
    h5_name = dan_h5_fname(dataset, mode)

    all_bg = _all_bg()
    dataset_root = os.path.join(DAN_ROOT, dataset)

    with h5py.File(h5_name, mode='a') as h5:
        root, dirs, _ = next(os.walk(dataset_root))

        speakers = defaultdict(list)
        # gather all speakers
        for _dir in dirs:
            _, wavs, _ = next(os.walk(dataset_root))
            for wav in [wav for wav in wavs if wav.lower().endswith('wav')]:
                spk = os.path.basename(wav).split('-')[0]
                speakers[spk].append(os.path.join(root, _dir, wav))

        speaker_stat = {}
        for (speaker, wavfiles) in tqdm(speakers, desc='spk'):
            # we take at most 10
            wavfiles = sorted(wavfiles)[:10]

            total_duration_sec = _process_speaker(
                speaker,
                wavfiles,
                h5,
                mode,
                aug_option={
                    'bgNoise': all_bg,
                    'overwrite': overwrite,
                })

            speaker_stat[speaker] = total_duration_sec

        glog.info('\n' * 10)
        glog.info(pformat(speaker_stat))


p = mp.Pool(mp.cpu_count() // 2)

if __name__ == '__main__':
    import fire

    fire.Fire({
        'vctk': vctk,
        'timit': timit,
        'voice': voice,
        'fff': fff_en,
        'ffh': ffh_jp,
        'bg': _all_bg,
    })
