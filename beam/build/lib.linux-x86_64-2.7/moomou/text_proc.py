# -*- coding: utf-8 -*-
from __future__ import absolute_import

import spacy
from spacy.tokens import Doc

class MetaChar(Enum):
    # Using special, one char symbol
    NUM = 'ø'
    NEW_LINE = 'π'
    SPACE = '†'

TXT_PIPE_STEPS = [
    (textacy.preprocess.normalize_whitespace, None),
    (textacy.preprocess.replace_numbers, [ str(MetaChar.NUM.value) ]),
    (replace_re_with, [ COMPILED_NEW_LINE_RE, str(MetaChar.NEW_LINE.value) ]),
]

def replace_re_with(text, regex, replacement=' '):
    return re.sub(regex, replacement, text)

def text_pipe(raw_txt, pipe=TXT_PIPE_STEPS):
    txt = raw_txt

    for func, params in pipe:
        if params is not None:
            txt = func(txt, *params)
        else:
            txt = func(txt)

    return txt
