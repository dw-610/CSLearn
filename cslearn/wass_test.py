"""
This (temporary) module contains code for facilitating testing of the 
Wasserstein loss training methods.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import tensorflow as tf

# ------------------------------------------------------------------------------

def get_data():
    """
    Function for loading in MNIST data.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255

    x_train = 2*x_train - 1
    x_test = 2*x_test - 1

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)