from enum import Enum     # for enum34, or the stdlib version

Lang = Enum(
    'Lang',

    # NOTE: Order is important - MUST NOT CHANGE, can only append
    ' '.join([
        'c',
        'cpp',
        'csharp',
        'css',
        'golang',
        'haskell',
        'html',
        'java',
        'javascript',
        'matlab',
        'objC',
        'perl',
        'php',
        'python',
        'r',
        'ruby',
        'scala',
        'shell',
        'sql',
        'swift',
    ])
)

LangByValue = {
    str(l.value): l for l in Lang
}
LangByName = {
    str(l.name): l for l in Lang
}

def get_type(path):
    if path.endswith('.cs'):
        return Lang.csharp

    if path.endswith('.js'):
        return Lang.javascript

    if path.endswith('.py'):
        return Lang.python

    if path.endswith('.php'):
        return Lang.php

    if path.endswith('.cpp') or path.endswith('.hpp'):
        return Lang.cpp

    if path.endswith('.c') or path.endswith('.h'):
        return Lang.c

    if path.endswith('.sql'):
        return Lang.sql

    if path.endswith('.sass') or path.endswith('.css'):
        return Lang.css

    if path.endswith('.rb'):
        return Lang.ruby

    if path.endswith('.html'):
        return Lang.html

    if path.endswith('.groovy') or path.endswith('.cli') or path.endswith('.java'):
        return Lang.java

    if path.endswith('.sh') or path.endswith('.bash'):
        return Lang.shell

    if path.endswith('.scala'):
        return Lang.scala

    if path.endswith('.hs'):
        return Lang.haskell

    if path.endswith('.swift'):
        return Lang.swift

    if path.endswith('.go'):
        return Lang.golang

    if path.endswith('.r'):
        return Lang.r

    if path.endswith('.pl'):
        return Lang.perl

    if path.endswith('.mm'):
        return Lang.objC

def get_type_by_value(value):
    return LangByValue[value]

def get_type_by_name(name):
    return LangByName[name]

import re
import textacy
import spacy
from spacy.tokens import Doc
from nltk.util import ngrams

ID_COL = 'F_id'
PATH_COL = 'F_path'
CONTENT_COL = 'C_content'

COMPILED_NEW_LINE_RE = re.compile('\n')
COMPILED_WS_RE = re.compile(' ')

def replace_re_with(text, regex, replacement=' '):
    return re.sub(regex, replacement, text)

class MetaChar(Enum):
    # Using special, one char symbol
    NUM = 'ø'
    NEW_LINE = 'π'
    SPACE = '†'

TXT_PIPE_STEPS = [
    (textacy.preprocess.normalize_whitespace, None),
    (textacy.preprocess.replace_numbers, [ str(MetaChar.NUM.value) ]),
    (replace_re_with, [ COMPILED_NEW_LINE_RE, str(MetaChar.NEW_LINE.value) ]),
    (ngrams, [ 3 ]),
]

def text_pipe(raw_txt, pipe=TXT_PIPE_STEPS):
    txt = raw_txt
    for func, params in pipe:
        if params is not None:
            txt = func(txt, *params)
        else:
            txt = func(txt)
    return txt

class PlangTokenizer(object):
    def __init__(self, nlp):
        self.vocab = nlp.vocab

    def __call__(self, text):
        seq = text_pipe(text)

        seq = [re.sub(COMPILED_WS_RE, str(MetaChar.SPACE.value), ''.join(n)) for n in seq]
        spaces = [True] * len(seq)

        return Doc(self.vocab, words=seq, spaces=spaces)
