"""
This module contains functions to implement custom training methods for the
various in the cslearn package.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from .arch import layers

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

def get_reconstruction_gradient(
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
    """
    This function computes the gradient of the image reconstruction loss term
    for the given true images and predictions.

    It is assumed that the reconstruction loss is the mean squared error.

    The resulting gradient is that of the loss with respect to the output image
    variables, which can the be used to backpropagate through the model.

    Parameters
    ----------
    y_true : tf.Tensor
        True images.
    y_pred : tf.Tensor
        Predicted images.

    Returns
    -------
    tf.Tensor
        The gradient of the image reconstruction loss. This is a tensor of the
        same shape as y_true and y_pred, i.e., it is the gradient of the loss
        for *each sample* in the batch (not aggregated).
    """
    return 2*(y_pred - y_true)

# ------------------------------------------------------------------------------

@tf.function
def wasserstein_domain_learner_train_step(
    model: tf.keras.models.Model, 
    optimizer: tf.keras.optimizers.Optimizer,
    batch_data: tf.Tensor,
    batch_labels: tf.Tensor,
    K_matrix: tf.Tensor
    ):
    """
    This function performs a training step over a single batch of data for a
    domain learner model with the custom Wasserstein loss.

    Parameters
    ----------
    model : cslearn.arch.models.DomainLearnerModel
        The model to train.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use.
    batch_data : tf.Tensor
        The input data for the batch.
    batch_labels : tf.Tensor
        The true labels for the batch.
    K_matrix : tf.Tensor
        Matrix exponential of -lam*M-1, where M is the ground metric matrix.
    """
    true_images = batch_labels[0]
    true_properties = batch_labels[1]
    with tf.GradientTape() as tape:
        pred_images, pred_properties, _ = model(batch_data, training=True)
        tape.watch(pred_images)
        tape.watch(pred_properties)
        dlw_dc = get_wasserstein_gradient(
            y_true=true_properties, 
            y_pred=pred_properties,
            K_matrix=K_matrix
        )
        dlr_dx = get_reconstruction_gradient(
            y_true=true_images,
            y_pred=pred_images
        )

    grads = tape.gradient(
        [pred_images, pred_properties],
        model.trainable_variables,
        output_gradients=[
            tf.multiply(model.alpha, dlr_dx), 
            tf.multiply(model.beta, dlw_dc)
        ]
    )
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return pred_images, pred_properties

# ------------------------------------------------------------------------------
    
@tf.function
def wasserstein_domain_learner_test_step(
    model: tf.keras.models.Model,
    test_data: tf.Tensor,
    test_labels: tf.Tensor
    ) -> tf.Tensor:
    """
    This function performs a testing step over some data for a domain learner
    model with the custom Wasserstein loss.

    Parameters
    ----------
    model : cslearn.arch.models.DomainLearnerModel
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
    true_images = test_labels[0]
    true_properties = test_labels[1]
    pred_images, pred_properties, _ = model(test_data, training=False)
    acc = tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.argmax(pred_properties, axis=-1),
                tf.argmax(true_properties, axis=-1)
            ),
            tf.float32
        )
    )
    lr = tf.reduce_mean(tf.square(pred_images - true_images))
    return acc, lr

# ------------------------------------------------------------------------------

def wasserstein_domain_learner_train_loop(
        model: tf.keras.models.Model,
        encoder: tf.keras.models.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        training_loader: tf.data.Dataset,
        validation_loader: tf.data.Dataset,
        epochs: int,
        batch_size: int,
        train_size: int,
        valid_size: int,
        warmup: int,
        mu: float,
        proto_update_step_size: int,
        number_of_properties: int,
        latent_dim: int
    ) -> dict:
    """
    Function to train a domain learner model with the custom Wasserstein loss.

    Makes use of the tqdm library to display a progress bar for the training
    loop.

    Parameters
    ----------
    model : cslearn.arch.models.DomainLearnerModel
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
    warmup : int, optional
        The number of epochs to train with only reconstruction loss. After this,
        the Wasserstein loss is added.
    """
    history = {'accuracy': [], 'lr': [], 'val_accuracy': [], 'val_lr': []}
    steps_per_epoch = train_size // batch_size + 1
    print('\nStarting training loop for domain learner with Wasserstein loss...')
    for epoch in range(epochs):
        print(f'\nEpoch: {epoch+1}/{epochs}')
        print(f'Learning rate: {optimizer.lr.numpy()}')

        # set the beta parameter based on the current epoch
        if epoch < warmup:
            model.beta.assign(0.0)
        elif epoch == warmup:
            model.beta.assign(model.beta_val)

        # update protos during warmup
        if (warmup > 0) and (epoch <= warmup) and (epoch > 0):
            new_protos = domain_learner_update_prototypes(
                old_prototypes=model.protos.numpy(),
                encoder=encoder,
                training_loader=training_loader,
                autoencoder_type='standard', # TODO: add VAE at some point
                mu=mu,
                proto_update_type='average',
                batches=proto_update_step_size,
                number_of_properties=number_of_properties,
                latent_dim=latent_dim,
                steps_per_epoch=steps_per_epoch,
                verbose=True
            )
            model.protos.assign(new_protos)

        # compute K_matrix
        # TODO: if no M given, dynamically update M with prototypes
        lam = model.wasserstein_lam
        M = model.metric_matrix**model.wasserstein_p
        K_matrix = tf.exp(-lam*M-1)

        # training
        acc = 0
        lr = 0
        for data, labels in tqdm(
            training_loader.take(steps_per_epoch),
            desc='Steps',
            ncols=80,
            total=train_size//batch_size+1
        ):
            pred_ims, pred_props = wasserstein_domain_learner_train_step(
                model=model,
                optimizer=optimizer,
                batch_data=data,
                batch_labels=labels,
                K_matrix=K_matrix
            )
            # get metrics for the current step
            acc += tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.argmax(pred_props, axis=-1),
                        tf.argmax(labels[1], axis=-1)
                    ),
                    tf.float32
                )
            )
            lr += tf.reduce_mean(tf.square(pred_ims - labels[0]))
        acc = acc.numpy()/steps_per_epoch
        lr = lr.numpy()/steps_per_epoch
        history['accuracy'].append(acc)
        history['lr'].append(lr)
        print(f'Training accuracy = {np.round(acc*100,2)}%')
        print(f'Training reconstruction loss = {np.round(lr,2)}')

        # validation
        print('Testing on the validation set...', end=' ')
        valid_steps = valid_size // batch_size + 1
        acc = 0
        lr = 0
        for data, labels in validation_loader.take(valid_steps):
            a, l = wasserstein_domain_learner_test_step(
                model=model,
                test_data=data,
                test_labels=labels
            )
            acc += a
            lr += l
        acc = acc.numpy()/valid_steps
        lr = lr.numpy()/valid_steps
        history['val_accuracy'].append(acc)
        history['val_lr'].append(lr)
        print(f'accuracy = {np.round(acc*100,2)}%')
        print(f'reconstruction loss = {np.round(lr,2)}')

        # update protos (when not in warmup stage)
        if epoch > warmup-1:
            new_protos = domain_learner_update_prototypes(
                old_prototypes=model.protos.numpy(),
                encoder=encoder,
                training_loader=training_loader,
                autoencoder_type='standard', # TODO: add VAE at some point
                mu=mu,
                proto_update_type='average',
                batches=proto_update_step_size,
                number_of_properties=number_of_properties,
                latent_dim=latent_dim,
                steps_per_epoch=steps_per_epoch,
                verbose=True
            )
            model.protos.assign(new_protos)

    return history

# ------------------------------------------------------------------------------

def domain_learner_update_prototypes(
        old_prototypes: np.ndarray,
        encoder: tf.keras.models.Model,
        training_loader: tf.data.Dataset,
        autoencoder_type: str,
        mu: float,
        proto_update_type: str,
        batches: int,
        number_of_properties: int,
        latent_dim: int,
        steps_per_epoch: int,
        verbose: bool = True,
    ):
    """
    Function to update the prototypes of a domain learner model based on the
    current state of the model.

    Parameters
    ----------
    mu : float
        The learning rate for the prototype update.
    proto_update_type : str
        The type of update to perform. Currently, only 'average' is supported.
    batches : int
        The number of batches to use for the update.
    verbose : bool, optional
        Whether to print feedback during the update.
        Default is True.
    """
    # get number of prototypes and features
    num_ps = number_of_properties
    num_fs = latent_dim

    # intialize arrays
    pred_props = np.zeros(shape=(0, num_ps))
    features_all = np.zeros(shape=(0, num_fs))

    # get number of batches if not specified (default to steps_per_epoch)
    if batches is None:
        batches = steps_per_epoch

    # if number of batches specified, loop through that number of batches
    i = 0
    for batch in training_loader.take(batches).as_numpy_iterator():
        inputs = batch[0]
        props = batch[1][1]
        features = encoder(inputs, training=False).numpy()
        if autoencoder_type == 'variational':
            features = layers.ReparameterizationLayer(
                latent_dim
            )(features)[2].numpy()

        pred_props = np.append(pred_props, values=props, axis=0)
        features_all = np.append(features_all, values=features, axis=0)

        if verbose:
            print(
                f'\rRecomputing prototypes... {i+1}/{batches}', 
                end = ''
            )
        i += 1

    # add a line after last print statement
    if verbose:
        print()

    # convert the outputs from one-hot to a prediction vector
    pred_props_vec = np.argmax(pred_props, axis=1)

    # loop through prototypes and recompute
    new_prototypes = np.copy(old_prototypes)
    for p in range(num_ps):
        # make sure that at least one p is present
        if np.any(pred_props_vec==p):
            p_feats = features_all[pred_props_vec==p,:]
            old_proto = old_prototypes[p,:]

            if proto_update_type == 'average':
                new_proto = np.mean(p_feats, axis=0)*(1-mu) + old_proto*mu
            else:
                raise ValueError("Invalid prototype update type.")
            
            # print feedback
            if verbose:
                print(f'Prototype {p+1}/{num_ps} updated.')

            # update the prototype
            new_prototypes[p,:] = new_proto

        # if no p is present, just keep the old prototype
        else:
            if verbose:
                print(f'WARNING: Prototype {p+1}/{num_ps} not updated.')

    return new_prototypes

# ------------------------------------------------------------------------------