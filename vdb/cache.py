import os
from constant import HOME_PREFIX

CACHE_DIR = os.path.join(HOME_PREFIX, '_cache')


def cache_run(key, fn):
    cached = os.path.join(CACHE_DIR, key)

    if os.path.exists(cached):
        return cached

    fn(cached)
    return cached
