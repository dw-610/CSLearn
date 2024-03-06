"""
This (temporary) module contains code for facilitating testing of the 
Wasserstein loss training methods.
"""

# ------------------------------------------------------------------------------
# imports

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from .arch.grads import wasserstein_grad

# ------------------------------------------------------------------------------

def get_data(dataset: str = 'mnist'):
    """
    Function for loading in MNIST or CIFAR-10 data.
    """

    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise ValueError('Invalid dataset name.')

    if dataset == 'mnist':
        x_train = np.expand_dims(x_train,-1).astype('float32') / 255
        x_test = np.expand_dims(x_test,-1).astype('float32') / 255
    else:
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

    x_train = 2*x_train - 1
    x_test = 2*x_test - 1

    y_train = tf.one_hot(np.squeeze(y_train), 10)
    y_test = tf.one_hot(np.squeeze(y_test), 10)

    return (x_train, y_train), (x_test, y_test)

# ------------------------------------------------------------------------------

def get_model(dataset: str = 'mnist'):
    """
    Function for creating a basic CNN model for MNIST or CIFAR-10.
    """

    if dataset == 'mnist':
        input_shape = (28, 28, 1)
    elif dataset == 'cifar10':
        input_shape = (32, 32, 3)
    else:
        raise ValueError('Invalid dataset name.')
    num_classes = 10

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3,3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    enc_out = tf.keras.layers.Dense(2, activation='linear')(x)

    encoder = tf.keras.models.Model(inputs=inputs, outputs=enc_out)

    x = encoder(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
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

def train_model(epochs, batch_size, model, optimizer, x_trn, y_trn, x_tst, y_tst, lam, Kmat):
    """
    Function for training the model with the custom Wasserstein loss.
    """
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        # b = 1
        for i in tqdm(range(0, len(x_trn), batch_size), desc='Batches',
                      ncols=80):
            if i+batch_size > len(x_trn):
                x_batch = x_trn[i:]
                y_batch = y_trn[i:]
            else:
                x_batch = x_trn[i:i+batch_size]
                y_batch = y_trn[i:i+batch_size]
            pred = train_step(model, optimizer, x_batch, y_batch, lam, Kmat)
            if tf.reduce_any(tf.math.is_nan(pred)):
                print('NaN prediction detected.')
                break
            # print(f'\rBatch {b}/{len(x_trn)//batch_size + 1}', end='')
            # b += 1
        # print()
        # print('Evaluating model...')
        acc = test_step(model, x_tst, y_tst)
        acc = np.round(acc.numpy()*100, 2)
        print(f'Validation accuracy: {acc:.2f}%')

# ------------------------------------------------------------------------------

def plot_features(features, labels, legend: list):
    """
    Function for plotting the 2D feature space.
    """
    import matplotlib.pyplot as plt

    labels = np.argmax(labels, axis=-1)

    for i in range(10):
        plt.scatter(
            features[labels==i,0],
            features[labels==i,1],
            s=10,
            label=legend[i]
        )
    plt.legend()
    plt.grid()
    plt.show()

# ------------------------------------------------------------------------------
    
def confirm_metric(M):
    dim = M.shape[0]

    # distance should be 0 iff the points are the same
    for i in range(dim):
        for j in range(dim):
            if i==j and M[i,i] != 0:
                raise ValueError('Diagonal elements should be 0.')
            if i!=j and M[i,j] == 0:
                raise ValueError('Off-diagonal elements should be non-zero.')
        
    # non-negativity and symmetry
    for i in range(dim):
        for j in range(dim):
            if M[i,j] < 0:
                raise ValueError('Non-negativity not satisfied.')
            if M[i,j] != M[j,i]:
                raise ValueError('Symmetry not satisfied.')

    # triangle inequality
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                    if M[i,j] + M[j,k] < M[i,k]:
                        raise ValueError('Triangle inequality not satisfied.')
    return True

# ------------------------------------------------------------------------------
