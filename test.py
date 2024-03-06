import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from cslearn import wass_test as wt

import sys

def main():
    # set parameters
    lam = 1.0
    p = 1.0

    # mnist
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
    ], dtype=tf.float32)

    # # cifar-10
    # M = tf.constant([
    #     [0, 2, 2, 3, 3, 3, 3, 3, 1, 2],
    #     [2, 0, 3, 2, 2, 2, 2, 2, 2, 1],
    #     [2, 3, 0, 1, 2, 1, 1, 2, 3, 3],
    #     [3, 2, 1, 0, 2, 1, 1, 2, 3, 3],
    #     [3, 2, 2, 2, 0, 2, 2, 1, 3, 3],
    #     [3, 2, 1, 1, 2, 0, 1, 2, 3, 3],
    #     [3, 2, 1, 1, 2, 1, 0, 2, 3, 3],
    #     [3, 2, 2, 2, 1, 2, 2, 0, 3, 3],
    #     [1, 2, 3, 3, 3, 3, 3, 3, 0, 2],
    #     [2, 1, 3, 3, 3, 3, 3, 3, 2, 0],
    # ], dtype=tf.float32)

    if not wt.confirm_metric(M):
        raise ValueError('The matrix M is not a metric.')

    M = M**p
    Kmat = tf.exp(-lam*M-1)

    dataset = 'mnist'

    # breakpoint()

    # get the data and model
    (x_trn, y_trn), (x_tst, y_tst) = wt.get_data(dataset)
    encoder, model = wt.get_model(dataset)
    encoder.summary()
    model.summary()

    # set the optimizer and compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)

    # train the model
    wt.train_model(
        epochs=20,
        batch_size=128,
        model=model,
        optimizer=optimizer,
        x_trn=x_trn,
        y_trn=y_trn,
        x_tst=x_tst,
        y_tst=y_tst,
        lam=lam,
        Kmat=Kmat
    )

    # evaluate the feature space
    legend = [str(i) for i in range(10)] # MNIST
    # legend = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
    #           'dog', 'frog', 'horse', 'ship', 'truck'] # CIFAR-10
    features = encoder(x_tst, training=False).numpy()
    wt.plot_features(features, y_tst, legend)

if __name__ == '__main__':
    main()