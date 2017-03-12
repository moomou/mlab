import re

from mpi4py import MPI
import h5py
import numpy as np

class Embedding():
    def __init__(self, embedding_h5):
        self.embedding = embedding_h5['embedding']

    def words_to_embeddings(self, words):
        words = [w.strip() for w in words.split(' ') if w.strip()]

        embeddings = np.zeros(
            (len(words), self.embedding['dimension'].value[0])
        )

        for idx, word in enumerate(words):
            try:
                emb = self.embedding.get(
                    re.escape(word),
                    self.embedding[re.escape('<unk>')]
                )
                embeddings[idx] = emb.value
            except:
                print(word)
                print(self.embedding.get(re.escape(word), '<unk>'))
                raise

        return embeddings
