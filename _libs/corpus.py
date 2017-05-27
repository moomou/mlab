#!/usr/bin/env python

import sys

from main import build_corpus

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage:: %s output prefix' % sys.argv[0])

    ofilename = sys.argv[1]
    prefix = sys.argv[2]

    files = ['%s/%s.csv' % (prefix, i) for i in range(0, 10)]
    build_corpus(ofilename, *files)
