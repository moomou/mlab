#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import random
import re

NEWLINE_SYMBOL = 'π'
UNKNOWN_SYMBOL = '÷'
# 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DEFAULT_CHAR_LIST = 'abcdefghijklmnopqrstuvwxyz0123456789,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{} π'
DEFAULT_CHAR_SET = set(DEFAULT_CHAR_LIST)
assert len(DEFAULT_CHAR_SET) == len(DEFAULT_CHAR_LIST)
DEFAULT_CHAR_IND = {char: index for index,
                    char in enumerate(DEFAULT_CHAR_LIST)}
WS_IND = DEFAULT_CHAR_IND[' ']

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

SNIPPET_LEN = 256


def _reservoir_sampling(items, k):
    '''
    Given a file f, uses reservoir sampling
    to return `sample_count` lines

    source::
    http://blog.cloudera.com/blog/2013/04/hadoop-stratified-randosampling-algorithm/
    http://www.geeksforgeeks.org/reservoir-sampling/
    '''
    samples = []
    for i, content in enumerate(items):
        if i < k:
            samples.append(content)
        else:
            j = random.randrange(1, i + 1)
            if j < k:
                samples[j] = content

    return samples


def sample_k_line_from_file(f, sample_count=1, snippet_mode=False):
    sampled = _reservoir_sampling(f, sample_count)
    if not snippet_mode:
        return sampled

    snippets = []
    for line in sampled:
        snippets.extend(break_line_2_snippets(line))

    return snippets


def break_line_2_snippets(line):
    snippets = []
    for i in range(0, len(line), SNIPPET_LEN):
        snippet = line[i: i + SNIPPET_LEN]
        snippets.append(snippet)

    return snippets


def sample_k_snippet_from_line(line, k=1):
    snippets = break_line_2_snippets(line)
    return _reservoir_sampling(snippets, k)


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


def doc_to_word_index(doc, max_len=None):
    if max_len is None:
        max_len = len(doc)

    doc_word_indices = np.zeros(max_len)
    doc_word_indices[:] = WS_IND

    for j, char in enumerate(doc[-max_len:]):
        char = char.lower()
        if char == '\n':
            char = NEWLINE_SYMBOL
        char_ind = DEFAULT_CHAR_IND.get(char, WS_IND)
        doc_word_indices[j] = char_ind

    return doc_word_indices


if __name__ == '__main__':
    lorem = 'Ipsum optio dolore vero in nihil aliquam libero aliquam voluptas! Cumque sunt tenetur maiores expedita vel repellendus optio nobis possimus exercitationem atque quis? Illo tempora aperiam nisi quasi accusantium labore.Elit deserunt ea nostrum molestias maxime veniam laboriosam. Tempore neque ab assumenda impedit recusandae vitae alias deleniti. Mollitia magni reiciendis sunt assumenda laboriosam. Possimus adipisicing dolorem suscipit voluptas ut! Eveniet?'
    print(len(sample_k_snippet_from_line(lorem)))
