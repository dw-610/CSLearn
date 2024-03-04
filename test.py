import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cslearn import wass_test as wt

import sys

def main():
    # set parameters
    lam = 1.0
    p = 1.0
    M = tf.constant([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        [3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
        [4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        [6, 5, 4, 3, 2, 1, 0, 1, 2, 3],
        [7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
        [8, 7, 6, 5, 4, 3, 2, 1, 0, 1],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    ], dtype=np.float32)
    M = M**p
    Kmat = tf.exp(-lam*M-1)

    # breakpoint()

    # get the data and model
    (x_trn, y_trn), (x_tst, y_tst) = wt.get_data()
    encoder, model = wt.get_model()
    encoder.summary()
    model.summary()

    # set the optimizer and compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)

    # train the model
    for epoch in range(30):
        print('\nEpoch: ', epoch+1, '/30')
        for i in range(0, len(x_trn), 64):
            x_batch = x_trn[i:i+32]
            y_batch = y_trn[i:i+32]
            pred = wt.train_step(model, optimizer, x_batch, y_batch, lam, Kmat)
            if tf.reduce_any(tf.math.is_nan(pred)):
                print('NaN prediction detected.')
                break
            print(f'\rBatch {i+1}/{len(x_trn)}', end='')
        print()
        print('Evaluating model...')
        acc = wt.test_step(model, x_tst, y_tst)
        acc = np.round(acc.numpy()*100, 2)
        print(f'Accuracy: {acc:.2f}%')

    # evaluate the feature space
    features = encoder(x_tst, training=False).numpy()
    plt.scatter(features[:,0], features[:,1], c=np.argmax(y_tst, axis=1))
    plt.show()

if __name__ == '__main__':
    main()