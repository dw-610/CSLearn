"""
This (temporary) module contains code for facilitating testing of the 
Wasserstein loss training methods.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import tensorflow as tf

from .arch.grads import wasserstein_grad

# ------------------------------------------------------------------------------

def get_data():
    """
    Function for loading in MNIST data.
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train,-1).astype('float32') / 255
    x_test = np.expand_dims(x_test,-1).astype('float32') / 255

    x_train = 2*x_train - 1
    x_test = 2*x_test - 1

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)

# ------------------------------------------------------------------------------

def get_model():
    """
    Function for creating a basic CNN model for MNIST.
    """

    input_shape = (28, 28, 1)
    num_classes = 10

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    enc_out = tf.keras.layers.Dense(2, activation='linear')(x)

    encoder = tf.keras.models.Model(inputs=inputs, outputs=enc_out)

    x = encoder(inputs)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return encoder, model

# ------------------------------------------------------------------------------

@tf.function
def train_step(model, optimizer, inputs, outputs, lam, Kmat):
    """
    Function for a single training step with the custom Wasserstein loss.
    """
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        tape.watch(predictions)
        # Directly compute dW/dh(x)
        dW_dh = wasserstein_grad(outputs, predictions, lam, Kmat)
    
    grads = tape.gradient(
        predictions,
        model.trainable_variables,
        output_gradients=dW_dh
    )

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return predictions

# ------------------------------------------------------------------------------

@tf.function
def test_step(model, inputs, outputs):
    """
    Function for a single testing step with the custom Wasserstein loss.
    """
    predictions = model(inputs, training=False)

    # compute accuracy
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(predictions, axis=-1),
                tf.argmax(outputs, axis=-1)
            ),
            tf.float32
        )
    )

    return acc

# ------------------------------------------------------------------------------

