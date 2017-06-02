#!/usr/bin/env python
import keras
from keras import optimizers

from data4 import (
    load_dataset,
    langs,
)
from plan5 import (
    CHAR_MAX_LEN,
    create_model,
)
from util import LossHistory

if __name__ == '__main__':
    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = load_dataset(CHAR_MAX_LEN)

    model = create_model(len(langs))
    model.summary()

    # opt = keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer='RMSprop', metrics=['accuracy'])

    with open('model.json', 'w') as f:
        f.write(model.to_json())

    file_name = 'plan5'
    check_cb = keras.callbacks.ModelCheckpoint(
        '/home/moomou/dev/mlab/plang_models/checkpoints/' + file_name +
        '.e{epoch:03d}-acc{val_acc:.2f}.hdf5',
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
        batch_size=512, epochs=512,
        shuffle=True,
        callbacks=[history, check_cb])

    print(history)
