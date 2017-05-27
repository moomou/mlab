# -*- coding: utf-8 -*-
import random
import re

import keras

NEWLINE_SYMBOL = 'π'
UNKNOWN_SYMBOL = '÷'
# 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DEFAULT_CHAR_LIST = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'"/\|_@#$%^&*~`+-=<>()[]{} π'
DEFAULT_CHAR_SET = set(DEFAULT_CHAR_LIST)
DEFAULT_CHAR_IND = {char: index for index,
                    char in enumerate(DEFAULT_CHAR_LIST)}

REMOVE_HASH_COMMENT = re.compile('.*#(.*)π')

DATA_DIR = '/home/moomou/dev/data/plang/raw'

# Order is paramount
LANGS = [
    'c',
    'cpp',
    'csharp',
    'css',
    'golang',
    'html',
    'java',
    'javascript',
    'perl',
    'php',
    'python',
    'ruby',
    'scala',
    'shell',
    'sql',
    'swift',
]

SNIPPET_LEN = 100


def _reservoir_sampling(items, k):
    '''
    Given a file f, uses reservoir sampling
    to return `sample_count` lines

    source::
    http://blog.cloudera.com/blog/2013/04/hadoop-stratified-randosampling-algorithm/
    http://www.geeksforgeeks.org/reservoir-sampling/
    '''
    sample = []
    for i, content in enumerate(items):
        if i <= k:
            sample.append(content)
        else:
            j = random.randrange(1, i + 1)
            if j <= k:
                sample[j] = content

    return sample


def sample_k_line_from_file(f, sample_count=1, snippet_mode=False):
    sampled = _reservoir_sampling(f, sample_count)
    if snippet_mode:
        return sampled

    snippets = []
    for line in snippets:
        snippets.extend(break_line_2_snippets(line))

    return snippets


def break_line_2_snippets(line):
    snippets = []
    for i in range(0, len(line), SNIPPET_LEN):
        snippets.append(line[i: i + SNIPPET_LEN])

    return snippets


def allow_only_char(txt, char_set=DEFAULT_CHAR_SET):
    chars = list(txt)
    cleaned = [c for c in chars if c in char_set]

    # if len(chars) != cleaned:
    # print('Sample::', chars, cleaned)

    return ''.join(cleaned)


def get_raw_data_filename(lang, index):
    return '-%s-%05d-of-00100' % (lang, index)


def get_data_filename(prefix, k, snippet_mode):
    return '%s_%s%s.txt' % (
        prefix,
        k,
        '_snippet' if snippet_mode else ''
    )


def remove_hash_comment(txt):
    return re.sub(REMOVE_HASH_COMMENT, '', txt)


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
