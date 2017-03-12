#!/usr/bin/env python
import sys

from mpi4py import MPI

from archiver import open_one
from lib import convert_to_word_embedding

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage:: %s filename output' % sys.argv[0])

    glove_h5 = sys.argv[1]
    prefix = sys.argv[2]
    inputs = sys.argv[3:]

    # The process ID (integer 0-3 for 4-process run)
    rank = MPI.COMM_WORLD.rank

    glove_h5 = open_one(glove_h5, mode='r', driver='mpio', comm=MPI.COMM_WORLD)
    convert_to_word_embedding(rank, glove_h5, prefix, *inputs)
    glove_h5.close()
