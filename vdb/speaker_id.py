import os

try:
    import cPickle as pickle
except:
    import pickle

HOME_PREFIX = '/home/moomou/dev/mlab/vdb/_db'
SPEAKER_COUNT_PATH = os.path.join(HOME_PREFIX, 'speaker_count.pickle')
SPEAKER_MAP_PATH = os.path.join(HOME_PREFIX, 'speaker_map.pickle')

counter = 0
_speaker_map_cache = {}


def _build_key(prefix, key):
    return '%s_%s' % (prefix, key)


def create_speaker_id(prefix, key):
    global _speaker_map_cache, counter

    cache_key = _build_key(prefix, key)

    if _speaker_map_cache.get(cache_key):
        return _speaker_map_cache[_build_key(prefix, key)]

    _speaker_map_cache[_build_key(prefix, key)] = counter
    speaker_id = counter
    counter += 1
    return speaker_id


def get_total_speaker():
    global counter
    return counter
