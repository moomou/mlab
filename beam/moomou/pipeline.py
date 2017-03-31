from __future__ import absolute_import

import os
import argparse
import logging
import re

import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.utils.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
)

from moomou.text_proc import text_pipe
from moomou.util import (
    CsvFileSource,
    plang,
)

def get_plang_tuple(row):
    _id, path, content = row

    if path.endswith('.cs'):
        return ('csharp', content)

    if path.endswith('.js'):
        return ('javascript', content)

    if path.endswith('.py'):
        return ('python', content)

    if path.endswith('.php'):
        return ('php', content)

    if path.endswith('.cpp') or path.endswith('.hpp'):
        return ('cpp', content)

    if path.endswith('.c') or path.endswith('.h'):
        return ('c', content)

    if path.endswith('.sql'):
        return ('sql', content)

    if path.endswith('.sass') or path.endswith('.css'):
        return ('css', content)

    if path.endswith('.rb'):
        return ('ruby', content)

    if path.endswith('.html'):
        return ('html', content)

    if path.endswith('.groovy') or path.endswith('.cli') or path.endswith('.java'):
        return ('java', content)

    if path.endswith('.sh') or path.endswith('.bash'):
        return ('shell', content)

    if path.endswith('.scala'):
        return ('scala', content)

    if path.endswith('.hs'):
        return ('haskell', content)

    if path.endswith('.swift'):
        return ('swift', content)

    if path.endswith('.go'):
        return ('golang', content)

    if path.endswith('.r'):
        return ('r', content)

    if path.endswith('.pl'):
        return ('perl', content)

    if path.endswith('.mm'):
        return ('objC', content)

    return None

def normalize_text(texts):
    docs = []
    for txt in texts:
        docs.append(text_pipe(txt))

    return '\n'.join(docs)

def merge_pcollection(pipe, prefix, limit):
    ps = []
    for i in range(int(limit)):
        str_i = str(i)
        filename = prefix + str_i if len(str_i) == 2 else prefix + '0' + str_i
        src = pipe | 'read[%d]' % i >> beam.io.Read(CsvFileSource(filename))
        ps.append(src)

    merged = tuple(ps) | 'flatten' >> beam.Flatten()
    return merged


def run(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--limit',
        dest='limit',
        default=2,
        help='Number of csv to read.')

    parser.add_argument('--input-prefix',
        dest='input_prefix',
        default='gs://dataflow-samples/shakespeare/kinglear.txt',
        help='Input prefix of files to process.')

    parser.add_argument('--output-prefix',
        dest='output_prefix',
        default='gs://YOUR_OUTPUT_BUCKET/AND_OUTPUT_PREFIX',
        help='Output file to write results to.')

    known_args, pipeline_args = parser.parse_known_args(argv)
    options = PipelineOptions()

    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    p = beam.Pipeline(options=pipeline_options)

    # Read the text file[pattern] into a PCollection.
    rows = merge_pcollection(p,
            known_args.input_prefix,
            int(known_args.limit))

    grouped = (rows
        | 'getPlangType' >> beam.Map(get_plang_tuple).with_output_types(beam.typehints.Tuple[str, str])
        | 'filtering' >> beam.Filter(lambda kv: kv)
        | 'groupByPlang' >> beam.GroupByKey().with_output_types(beam.typehints.Tuple[str, beam.typehints.Iterable[str]]))

    pcols = []
    for lang in plang:
        pcols.append(grouped
            | 'filter by %s' % lang >> beam.Filter(lambda kv, lang: kv[0] == lang, lang)
            | 'clean up %s' % lang >> beam.Map(lambda kv: normalize_text(kv[1]))
            | 'output %s' % lang >> beam.io.WriteToText(known_args.output_prefix + '-' + lang))

    p.run().wait_until_finish()

if '__main__' == __name__:
    logging.getLogger().setLevel(logging.INFO)
    # run()
