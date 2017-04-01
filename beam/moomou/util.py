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

    filtered = (line.replace('\0','') for line in self._file)
    reader = csv.reader(filtered)

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
