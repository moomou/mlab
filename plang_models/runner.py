#!/usr/bin/env python
import keras
from keras import optimizers

from data4 import (
    load_dataset,
    langs,
)
from plan6 import (
    NAME,
    CHAR_MAX_LEN,
    create_model,
)
from keras_util import LossHistory

if __name__ == '__main__':
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_dataset(max_len=CHAR_MAX_LEN, k=2000)

    model = create_model(len(langs))
    model.summary()

    # opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop', metrics=['accuracy'])

    with open('model_%s.json' % NAME, 'w') as f:
        f.write(model.to_json())

    check_cb = keras.callbacks.ModelCheckpoint(
        '/home/moomou/dev/mlab/plang_models/checkpoints/' + NAME +
        '.e{epoch:03d}-acc{val_acc:.3f}.hdf5',
        monitor='val_acc',
        verbose=0,
        save_best_only=True,
        mode='max'
    )

    early_stop_cb = keras.callbacks.EarlyStopping(
        mode='auto',
        monitor='val_loss',
        patience=5,
        verbose=0)

    history = LossHistory()

    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=256, epochs=256,
        shuffle=True,
        callbacks=[history, check_cb])

    print(history)
