# -*- coding: utf-8 -*-
from __future__ import absolute_import

import re
import textacy

COMMENT_RE = re.compile('\/\*[\s\S]*?\*\/|\/\/.*')
NEW_LINE_RE = re.compile('\n')
WS_RE = re.compile(' ')

class MetaChar():
    # Using special, one char symbol
    NUM = 'ø'
    NEW_LINE = 'π'
    SPACE = '†'

def replace_re_with(text, regex, replacement=' '):
    return re.sub(regex, replacement, text)

TXT_PIPE_STEPS = [
    (textacy.preprocess.normalize_whitespace, None),
    (replace_re_with, [ COMMENT_RE, '' ]),
    (textacy.preprocess.normalize_whitespace, None),
    (textacy.preprocess.replace_numbers, [ MetaChar.NUM.decode('utf-8') ]),
    (textacy.preprocess.normalize_whitespace, None),
    (replace_re_with, [ NEW_LINE_RE, MetaChar.NEW_LINE.decode('utf-8') ]),
]

def text_pipe(raw_txt, pipe=TXT_PIPE_STEPS):
    txt = raw_txt.decode('utf-8')

    for func, params in pipe:
        if params is not None:
            txt = func(txt, *params)
        else:
            txt = func(txt)

    return txt
