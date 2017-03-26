from __future__ import absolute_import

import argparse
import csv
import logging
import re
import sys

import apache_beam as beam

csv.field_size_limit(sys.maxsize)

class CsvFileSource(beam.io.filebasedsource.FileBasedSource):
  def read_records(self, file_name, range_tracker):
    self._file = self.open_file(file_name)

    reader = csv.reader(self._file)

    for rec in reader:
      yield rec

plang = [
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
]

def get_plang_type(path):
    if path.endswith('.cs'):
        return 'csharp'

    if path.endswith('.js'):
        return 'javascript'

    if path.endswith('.py'):
        return 'python'

    if path.endswith('.php'):
        return 'php'

    if path.endswith('.cpp') or path.endswith('.hpp'):
        return 'cpp'

    if path.endswith('.c') or path.endswith('.h'):
        return 'c'

    if path.endswith('.sql'):
        return 'sql'

    if path.endswith('.sass') or path.endswith('.css'):
        return 'css'

    if path.endswith('.rb'):
        return 'ruby'

    if path.endswith('.html'):
        return 'html'

    if path.endswith('.groovy') or path.endswith('.cli') or path.endswith('.java'):
        return 'java'

    if path.endswith('.sh') or path.endswith('.bash'):
        return 'shell'

    if path.endswith('.scala'):
        return 'scala'

    if path.endswith('.hs'):
        return 'haskell'

    if path.endswith('.swift'):
        return 'swift'

    if path.endswith('.go'):
        return 'golang'

    if path.endswith('.r'):
        return 'r'

    if path.endswith('.pl'):
        return 'perl'

    if path.endswith('.mm'):
        return 'objC'

    return None
