#!/usr/bin/env python3
import pickle

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA, IncrementalPCA
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib.colors import Normalize
from tqdm import tqdm


def tsne(X, y=None, dim=2, perplexity=65, generate_png=False):
    t = TSNE(n_components=dim, perplexity=perplexity, verbose=1, n_jobs=4)
    X_tsne = t.fit_transform(X)

    if generate_png:
        y_min, y_max = np.min(y), np.max(y)
        cmap = cm.Set1
        norm = Normalize(vmin=y_min, vmax=y_max)
        colors = [cmap(norm(i)) for i in y]

        plt.figure(figsize=(3.841, 7.195), dpi=100)
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)

        step_y = abs(np.max(X_tsne[:, 1]) - np.min(X_tsne[:, 1])) / 100
        step_x = abs(np.max(X_tsne[:, 0]) - np.min(X_tsne[:, 0])) / 100

        plt.yticks(
            np.arange(
                np.min(X_tsne[:, 1]) - step_y,
                np.max(X_tsne[:, 1]) + step_y, step_y))
        plt.xticks(
            np.arange(
                np.min(X_tsne[:, 0]) - step_x,
                np.max(X_tsne[:, 0]) + step_x, step_x))

        plt.ylabel('tsne - Y')
        plt.xlabel('tsne - X')
        plt.title('Tsne: p%d' % perplexity)

        ax = plt.gca()
        for i, txt in enumerate(X_tsne[:, 0]):
            ax.annotate(str(y[i]), (X_tsne[i, 0], X_tsne[i, 1]), fontsize=9)

        plt.savefig('output.svg', dpi=1200)

    return X_tsne


def plot_tsne_from_h5(h5, x_grp, y_grp):
    with h5py.File(h5) as f:
        X = []
        Y = []
        for key in f[x_grp]:
            X.append(f[x_grp][key])
        for key in f[y_grp]:
            Y.append(f[y_grp][key])

        tsne(X, Y, generate_png=True)


def pca_tsne(h5, x_grp, y_grp, pca_comp):
    with h5py.File(h5) as f:
        item_count = int(len(f[x_grp].keys()) * 0.3)

        # 2 ** 11 is the dimension of the embedding
        # 256 is the batch size
        X = np.zeros((item_count * 256, 2**11))
        Y = np.zeros((item_count * 256))

        for i, key in tqdm(enumerate(f[x_grp].keys()), 'load'):
            if i + 256 >= item_count:
                break

            X[i:i + 256, :] = f[x_grp][key]
            Y[i:i + 256] = f[y_grp][key]

        print('loaded into memory')
        pca = PCA(int(pca_comp), whiten=True)
        print('pca training...')
        pca.fit(X)

        print('variance:', pca.explained_variance_)
        print('variance_ratio:', pca.explained_variance_ratio_)

        pca_filename = 'pca_%s.output' % pca_comp
        with open(pca_filename, 'wb') as f:
            print('saved pca to `%s`' % pca_filename)
            pickle.dump(pca, f)

        pca_X = pca.transform(X)

        del pca
        del X
        tsne(pca_X, Y, generate_png=True)


if __name__ == '__main__':

    def test_tsne(seed=123, nb=1000, d=64):
        np.random.seed(seed)
        X = np.random.random((nb, d)).astype('float32')
        y = np.random.random_integers(100, size=(nb, 1))[:, 0]

        tsne(X, y, generate_png=True)

    import fire
    fire.Fire({
        't': test_tsne,
        't_from_h5': plot_tsne_from_h5,
        'pca_tsne': pca_tsne,
    })
