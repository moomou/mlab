#!/usr/bin/env python
import sys

from main import convert_to_word_embedding

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage:: %s filename output' % sys.argv[0])

    glove_h5 = sys.argv[1]
    prefix = sys.argv[2]
    inputs = sys.argv[3:]

    convert_to_word_embedding(glove_h5, prefix, *inputs)
