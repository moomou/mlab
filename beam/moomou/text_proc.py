# -*- coding: utf-8 -*-
import re

COMMENT_RE = re.compile('\/\*[\s\S]*?\*\/|\/\/.*')
NUM_RE = re.compile(r'(?:^|(?<=[^\w,.]))[+–-]?(([1-9]\d{0,2}(,\d{3})+(\.\d*)?)|([1-9]\d{0,2}([ .]\d{3})+(,\d*)?)|(\d*?[.,]\d+)|\d+)(?:$|(?=\b))')

NONBREAKING_SPACE_REGEX = re.compile(r'(?!\n)\s+')
LINEBREAK_REGEX = re.compile(r'((\r\n)|[\n\v])+')

NEW_LINE_RE = re.compile('\n')
WS_RE = re.compile(' ')

ALLOWED_CHAR = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:"\'/\|_@#$%^&*~+-=<>()[]{} '

class MetaChar():
    # Using special, one char symbol
    NEW_LINE = 'π'

def replace_re_with(text, regex, replacement=' '):
    return re.sub(regex, replacement, text)

def normalize_whitespace(text):
    return NONBREAKING_SPACE_REGEX.sub(' ', LINEBREAK_REGEX.sub(r'\n', text)).strip()

TXT_PIPE_STEPS = [
    (normalize_whitespace, None),
    (replace_re_with, [ COMMENT_RE, '' ]),
    (normalize_whitespace, None),
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
