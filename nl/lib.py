#!/usr/bin/env python
import csv
import math
import os
import re
import sys
import signal
import time

from mpi4py import MPI
import numpy as np
import spacy
import textacy

from embedding import Embedding
from plang_type import (
    CONTENT_COL,
    ID_COL,
    PATH_COL,

    Lang,
    MetaChar,
    PlangTokenizer,

    get_type,
    get_type_by_name,
)
from signal_wrapper import GracefulInterruptHandler
import archiver

NL_BATCH_SIZE = 300
CSV_BATCH_SIZE = NL_BATCH_SIZE * 5
THREAD = 4

csv.field_size_limit(sys.maxsize)

def _process(rows, nlp, writer):
    texts = (row[CONTENT_COL] for row in rows)

    start = time.time()
    print('Batching...', flush=True)

    for idx, doc in enumerate(nlp.pipe(texts, batch_size=NL_BATCH_SIZE, n_threads=THREAD)):
        rows[idx][CONTENT_COL] = doc.text
        try:
            rows[idx][PATH_COL] = get_type(rows[idx][PATH_COL]).name
        except:
            print('failed on ::', rows[idx][PATH_COL])
            raise

    for row in rows:
        writer.writerow(row)

    print('Committed:: %s' % (time.time() - start), flush=True)

def _plang_h5(prefix, mode='w', driver=None, comm=None):
    filenames = sorted(['%s%s.h5' % (prefix, e.name) for e in Lang])
    return archiver.open_multi(
        filenames, mode=mode, driver=driver, comm=comm)

def _plang_iterator(files):
    return { f: iter(f) for f in files.keys() }

def tokenize(ifilename, ofilename):
    fieldnames = [
        ID_COL,
        PATH_COL,
        CONTENT_COL,
    ]

    csvfile = open(ifilename, 'r', encoding='utf-8')
    filtered = (line.replace('\0','') for line in csvfile)
    reader = csv.DictReader(filtered, fieldnames=fieldnames)
    next(reader) # skip header row

    csvfile = open(ofilename, 'w', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    print('Starting on %s...' % os.path.basename(ifilename))
    nlp = spacy.load('en', create_make_doc=PlangTokenizer)
    rows = []

    for row in reader:
        rows.append(row)

        if len(rows) % CSV_BATCH_SIZE == 0:
            _process(rows, nlp, writer)
            rows = []

    if len(rows):
        _process(rows, nlp, writer)

def build_corpus(output, *files):
    total = len(files)
    counter = 0

    with open(output, 'wb+') as out:
        for filename in files:
            with open(filename, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    out.write((' ' + row[CONTENT_COL]).encode('utf-8'))
            counter += 1
            print('%s/%s' % (counter, total))

    print('Done!!')

def save_glove_h5py(glove_input, dim, output):
    out = archiver.open_one(output)

    embedding_group = out.create_group('embedding')
    embedding_group.create_dataset('dimension', data=[int(dim)], dtype='i')

    word_index = 0
    with open(glove_input) as f:
        for line in f:
            word, *vec = line.split(' ')
            vec = np.array(vec).astype('float')

            try:
                embedding_group.create_dataset(
                    re.escape(word),
                    data=vec,
                    dtype=vec.dtype
                )
            except:
                print(re.escape(word))
                raise

            word_index += 1

    embedding_group.create_dataset(
        'vocab_size',
        data=[word_index + 1],
        dtype='i')

def convert_to_word_embedding(rank, glove_h5, prefix, *inputs):
    files = _plang_h5(prefix, driver='mpio', comm=MPI.COMM_WORLD)
    embedding = Embedding(glove_h5)

    MAX_COUNTER = 900000
    files_counter = { key: 0 for key in files.keys() }

    print('[%d] Handling:: %s' % (rank, inputs), flush=True)

    with GracefulInterruptHandler() as h:
        for counter, f in enumerate(inputs):
            with open(f) as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    if h.interrupted: break

                    plang_enum = get_type_by_name(row[PATH_COL])

                    key = prefix + plang_enum.name + '.h5'
                    h5 = files[key]

                    if files_counter[key] > MAX_COUNTER:
                        continue
                    if files_counter[key] % 1000 == 0:
                        print('[%d] %s:: %d' % (rank, key, files_counter[key]), flush=True)

                    idx = '/%s' % row[ID_COL]
                    if idx not in h5:
                        files_counter[key] += 1

                        content = row[CONTENT_COL]

                        content_embedding = embedding.words_to_embeddings(
                            content)

                        h5.create_dataset(
                            idx,
                            data=content_embedding,
                            dtype=content_embedding.dtype
                        )

            print('[%d] finished %s' % (rank, counter + 1), flush=True)
            if h.interrupted: break

    print('[%d] exiting' % rank, flush=True)
    archiver.close_multi(files)

def compile_train_test_data(prefix):
    # size is file count
    train_size = int(1e7)
    test_size = int(2e6)

    # this is no. of triagrams that make up an input
    # TODO: shitty way to segmenting data, is there a better way??
    trigram_group_size = 24
    category_size = 20

    f_train_X = h5py.File('train_X.h5', 'w', chunks=True)
    train_X = f_train_X.create_dataset('data', (train_size, trigram_group_size, 50), dtype='float')

    f_train_y = h5py.File('train_y.h5', 'w', chunks=True)
    train_y = f_train_y.create_dataset('data', (train_size, 1, category_size), dtype='int')

    f_test_X = h5py.File('test_X.h5', 'w', chunks=True)
    test_X = f_test_X.create_dataset('data', (test_size, trigram_group_size, 50), dtype='float')

    f_test_y = h5py.File('test_y.h5', 'w', chunks=True)
    test_y = f_test_y.create_dataset('data', (test_size, 1, category_size), dtype='int')

    files = _plang_h5(prefix, mode='r')
    files_iterator = _plang_iterator(files)

    train_size_counter = 0
    while train_size_counter < train_size:
        for f in files.keys():
            p, f_type = f.split('_')

            assert p == 't', 'error'
            assert f_type.endswith('.h5'), 'error'

            f_type = f_type[:-3]

            # skip matlab
            if f_type.startswith('matlab'):
                continue

            try:
               dataset_name = next(files_iterator(f))

               doc = files[f][dataset_name]
               doc_trigram_counts = doc.shape[0]

               group_count = math.floor(doc_trigram_counts / trigram_group_size)
               start = 0

               for g in range(group_count):
                   start_idx = g * group_count
                   train_X[train_size_counter] = doc[start_idx:start_idx+ group_count]

                   ans = np.zeros((1, category_size))
                   ans[get_type_by_name(f_type).value] = 1
                   train_y[train_size_counter] = ans

                   train_size_counter += 1

            except StopIteration:
                pass

    test_size_counter = 0
    while test_size_counter < test_size:
        for f in files.keys():
            p, f_type = f.split('_')

            assert p == 't', 'error'
            assert f_type.endswith('.h5'), 'error'

            f_type = f_type[:-3]

            if f_type.startswith('matlab'):
                continue

            try:
               dataset_name = next(files_iterator(f))

               doc = files[f][dataset_name]
               doc_trigram_counts = doc.shape[0]

               group_count = math.floor(doc_trigram_counts / trigram_group_size)
               start = 0

               for g in range(group_count):
                   start_idx = g * group_count
                   test_X[test_size_counter] = doc[start_idx:start_idx+ group_count]

                   ans = np.zeros((1, category_size))
                   ans[get_type_by_name(f_type).value] = 1
                   test_y[test_size_counter] = ans

                   test_size_counter += 1

            except StopIteration:
                pass

    train_X.close()
    train_y.close()

    test_X.close()
    test_y.close()

    archiver.close_multi(files)

def raw_txt_to_embedding(embedding_file, content):
    # load nlp object
    nlp = spacy.load('en', create_make_doc=PlangTokenizer)
    embedding = Embedding(embedding_file)

    # push through it the text line pipe
    tokenized = nlp(content)

    # convert the trigrams to embedding
    content_embedding = embedding.words_to_embeddings(tokenized)

    # TODO: pipe to this to a keras model
    print(content_embedding)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage:: %s filename output' % sys.argv[0])

    ifilename = sys.argv[1]
    ofilename = sys.argv[2]

    tokenize(ifilename, ofilename)
