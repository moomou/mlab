#!/usr/bin/env python3

import h5py


def transfer_weight(model, weight_h5):
    for k in range(weight_h5.attrs['nb_layers']):
        if False and k >= len(model.layers):
            break

        g = weight_h5['layer_{}'.format(k)]
        weights = [
            g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])
        ]

        print('www::', weights)

        if model:
            model.layers[k].set_weights(weights)


def main(path):
    with h5py.File(path) as f:
        for l in f['model_weights'].keys():
            print(f['model_weights'].attrs['layer_names'])

        print(list(f.attrs.keys()))
        # transfer_weight(None, f)


if __name__ == '__main__':
    import fire
    fire.Fire({
        't': main,
    })
