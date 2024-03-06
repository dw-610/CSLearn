"""
This module contains functions to implement custom training methods for the
various in the cslearn package.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
import numpy as np

from tqdm import tqdm

# ------------------------------------------------------------------------------

def get_wasserstein_gradient(
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        K_matrix: tf.Tensor,
        lam: float = 1.0,
    ) -> tf.Tensor:
    """
    Function computing the gradient of the Wasserstein loss for the given true
    labels and predictions.

    The method used here is based on "Learning with a Wasserstein Loss" by
    Frogner et al. (2015), specifically Algorithm 1. Specifically, Sinkhorn
    iterations are used to compute the gradient.

    Parameters
    ----------
    y_true : tf.Tensor
        True labels, represented as a (batch of) histogram vector(s), e.g., the
        output of the softmax function.
    y_pred : tf.Tensor
        A (batch of) predicted label(s), represented as a histogram vector(s).
    K_matrix : tf.Tensor
        Matrix exponential of -lam*M-1, where M is the ground metric matrix.
    lam : float, optional
        Balancing parameter between the true total cost and entropic
        regularization.
        Default is 1.0.

    Returns
    -------
    tf.Tensor
        The gradient of the Wasserstein loss.
        This is a tensor of the same shape as y_true and y_pred, i.e., it is the
        gradient of the loss for *each sample* in the batch (not aggregated).
    """
    dim = y_true.shape[1]
    u = tf.ones_like(y_true)

    diff = tf.constant(float('inf'), dtype=tf.float32)
    while diff > 1e-3:
        u_ = u
        x = tf.matmul(u, K_matrix)
        x = tf.divide(y_true, x + 1e-16)
        x = tf.matmul(x, tf.transpose(K_matrix))
        u = tf.divide(y_pred, x + 1e-16) + 1e-16
        diff = tf.reduce_max(tf.reduce_sum(tf.square(u - u_),axis=-1))

    term1 = tf.math.log(u)/lam
    term2 = tf.repeat(
        tf.math.log(tf.reduce_sum(u, axis=1, keepdims=True))/dim/lam,
        dim,
        axis=1
    )

    grad = term1 + term2
    return grad

# ------------------------------------------------------------------------------

@tf.function
def wasserstein_classifier_train_step(
        model: tf.keras.models.Model, 
        optimizer: tf.keras.optimizers.Optimizer,
        batch_data: tf.Tensor,
        batch_labels: tf.Tensor,
        K_matrix: tf.Tensor,
        lam: float = 1.0
    ) -> tf.Tensor:
    """
    Function performing a training step over a single batch of data for a
    classifier model with the cutsom Wasserstein loss.

    Parameters
    ----------
    model : cslearn.arch.models.Classifier
        The model to train.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use.
    batch_data : tf.Tensor
        The input data for the batch.
    batch_labels : tf.Tensor
        The true labels for the batch.

    Returns
    -------
    tf.Tensor
        The predictions of the model on the batch.
    """
    with tf.GradientTape() as tape:
        predictions = model(batch_data, training=True)
        tape.watch(predictions)
        dW_dh = get_wasserstein_gradient(
            y_true=batch_labels, 
            y_pred=predictions,
            K_matrix=K_matrix,
            lam=lam
        )

    grads = tape.gradient(predictions, model.trainable_variables, output_gradients=dW_dh)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return predictions

# ------------------------------------------------------------------------------

@tf.function
def wasserstein_classifier_test_step(
        model: tf.keras.models.Model,
        test_data: tf.Tensor,
        test_labels: tf.Tensor
    ) -> tf.Tensor:
    """
    Function performing a testing step over some data for a classifier model
    with the custom Wasserstein loss.

    Parameters
    ----------
    model : cslearn.arch.models.Classifier
        The model to test.
    test_data : tf.Tensor
        The input data for the test.
    test_labels : tf.Tensor
        The true labels for the test data.

    Returns
    -------
    tf.Tensor
        The accuracy of the model on the test data.
    """
    predictions = model(test_data, training=False)
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(predictions, axis=-1),
                tf.argmax(test_labels, axis=-1)
            ),
            tf.float32
        )
    )
    return acc

# ------------------------------------------------------------------------------

def wasserstein_classifier_train_loop(
        model: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        training_loader: tf.data.Dataset,
        validation_loader: tf.data.Dataset,
        epochs: int,
        batch_size: int,
        train_size: int,
        valid_size: int,
    ) -> dict:
    """
    Function to train a classifier model with the custom Wasserstein loss.

    Makes use of the tqdm library to display a progress bar for the training
    loop.

    Parameters
    ----------
    model : cslearn.arch.models.Classifier
        The model to train.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use.
    training_loader : tf.data.Dataset
        The training data loader.
        Data samples should be images, and labels should be one-hot encoded.
    validation_loader : tf.data.Dataset
        The validation data loader.
        Data samples should be images, and labels should be one-hot encoded.
    epochs : int
        The number of epochs to train for.
    batch_size : int
        The batch size to use.
    train_size : int
        The number of samples in the training set.
    valid_size : int
        The number of samples in the validation set.
    """
    history = {'accuracy': [], 'val_accuracy': []}
    steps_per_epoch = train_size // batch_size + 1
    print('\nStarting training loop for classifier with Wasserstein loss...')
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        print(f'Learning rate: {optimizer.lr.numpy()}')
        # compute K_matrix
        lam = model.wasserstein_lam
        M = model.metric_matrix**model.wasserstein_p
        K_matrix = tf.exp(-lam*M-1)
        # training
        acc = 0
        for data, labels in tqdm(
            training_loader.take(steps_per_epoch),
            desc='Steps',
            ncols=80,
            total=train_size//batch_size+1
        ):
            preds = wasserstein_classifier_train_step(
                model=model,
                optimizer=optimizer,
                batch_data=data,
                batch_labels=labels,
                K_matrix=K_matrix,
                lam=lam
            )
            # compute the accuracy of the current step
            acc += tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(preds, axis=-1),
                        tf.argmax(labels, axis=-1)
                    ),
                    tf.float32
                )
            )
        acc = acc.numpy()/steps_per_epoch
        history['accuracy'].append(acc)
        print(f'Training accuracy = {np.round(acc*100,2)}%')
        # validation
        print('Testing on the validation set...', end=' ')
        valid_steps = valid_size // batch_size + 1
        acc = 0
        for data, labels in validation_loader.take(valid_steps):
            acc += wasserstein_classifier_test_step(
                model=model,
                test_data=data,
                test_labels=labels
            )
        acc = acc.numpy()/valid_steps
        history['val_accuracy'].append(acc)
        print(f'accuracy = {np.round(acc*100,2)}%')

    return history

# ------------------------------------------------------------------------------

