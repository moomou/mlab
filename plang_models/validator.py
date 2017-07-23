#!/usr/bin/env python
from keras.models import model_from_json
import glob
import os

from data4 import (
    load_dataset,
    convert_to_k_snippet,
)
from plan6 import (
    CHAR_MAX_LEN,
    AttentionWithContext,
)

if __name__ == '__main__':
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_dataset(max_len=CHAR_MAX_LEN, k=1, test_mode=True)

    # X_test, y_test = convert_to_k_snippet(X_test, y_test)
    for f in glob.glob('./model_json/*'):
        model = None

        with open(f) as config:
            model = model_from_json(config.read(), {
                'AttentionWithContext': AttentionWithContext,
            })

        model.summary()

        base = os.path.basename(f)
        prefix = base[len('model_'):-len('.json')]
        candidates = glob.glob('./checkpoints/%s*' % prefix)
        print('Candidates:: ', prefix, candidates)
        assert len(candidates) == 1, 'Too many weight for %s' % base

        model.load_weights(candidates[0])
        model.compile(loss='categorical_crossentropy',
                      optimizer='RMSprop', metrics=['accuracy'])

        print('Variant::', f)

        # estimate accuracy on whole dataset using loaded weights
        scores = model.evaluate(X_train, y_train, verbose=0)
        print('Scores::', scores)
        print("%s: %.3f%%" % (model.metrics_names[1], scores[1] * 100))
