#!/usr/bin/env python
from data4 import (
    load_dataset,
    langs,
)
from plan4 import (
    CHAR_MAX_LEN,
    create_model,
)

if __name__ == '__main__':
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_dataset(CHAR_MAX_LEN, k=2000, snippet_mode=True)

    model = create_model(len(langs))
    model.summary()

    model.load_weights('./checkpoints/plan4.e24-vl0.19.hdf5')

    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop', metrics=['accuracy'])

    # estimate accuracy on whole dataset using loaded weights
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Scores::', scores)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
