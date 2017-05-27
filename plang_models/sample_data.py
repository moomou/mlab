#!/usr/bin/env python
import os
import re

import numpy as np
import pandas as pd

from constant import (
    DATA_DIR
)

def striphtml(s):
    p = re.compile(r'<.*?>')
    return p.sub('', s)


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


def load_dataset(max_len, max_sent):
    data = pd.read_csv(
        os.path.join(DATA_DIR, 'labeledTrainData.tsv'),
        header=0, delimiter="\t", quoting=3)

    txt = ''
    docs = []
    sentences = []
    sentiments = []

    for cont, sentiment in zip(data.review, data.sentiment):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(striphtml(cont)))
        sentences = [sent.lower() for sent in sentences]
        docs.append(sentences)
        sentiments.append(sentiment)

    num_sent = []
    for doc in docs:
        num_sent.append(len(doc))
        for s in doc:
            txt += s

    chars = set(txt)

    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    print('Sample doc{}'.format(docs[1200]))

    X = np.ones((len(docs), max_sent, max_len), dtype=np.int64) * -1
    y = np.array(sentiments)

    for i, doc in enumerate(docs):
        for j, sentence in enumerate(doc):
            if j < max_sent:
                for t, char in enumerate(sentence[-max_len:]):
                    X[i, j, (max_len - 1 - t)] = char_indices[char]

    print('Sample X:{}'.format(X[1200, 2]))
    print('y:{}'.format(y[1200]))

    ids = np.arange(len(X))
    np.random.shuffle(ids)

    # shuffle
    X = X[ids]
    y = y[ids]

    X_train = X[:20000]
    X_test = X[22500:]

    y_train = y[:20000]
    y_test = y[22500:]

    return (
        X_train,
        y_train,
        X_test,
        y_test,
    )


if __name__ == '__main__':
    load_dataset(512, 15)
