"""
This module contains custom loss functions for us in the cslearn package.
"""

# ------------------------------------------------------------------------------
# imports

import tensorflow as tf

# ------------------------------------------------------------------------------

def wasserstein_grad(y_true, y_pred, lam: float = 1.0, Kmat: tf.Tensor = None):
    """
    Function computing the necessary gradient of the Wasserstein loss.

    Assumes that the underlying "ground metric" on the space of labels is the
    Euclidean distance.

    The method used here is based on "Learning with a Wasserstein Loss" by
    Frogner et al. (2015), specifically Algorithm 1.

    Parameters
    ----------
    y_true : tf.Tensor
        True labels, represented as a (batch of) histogram vector(s), e.g., the
        output of the softmax function.
    y_pred : tf.Tensor
        A (batch of) predicted label(s), represented as a histogram vector(s).
    lam : float
        Balancing parameter between the true total cost and entropic
        regularization.
    Kmat : tf.Tensor
        Matrix exponential of -lam*M-1, where M is the ground metric matrix.
    """
    if Kmat is None:
        raise ValueError('Kmat must be provided.')
    
    dim = y_true.shape[1]
    
    u = tf.ones_like(y_true)

    diff = tf.constant(float('inf'), dtype=tf.float32)

    while diff > 1e-3:
        u_ = u
        x = tf.matmul(u, Kmat)
        x = tf.divide(y_true, x + 1e-16)
        x = tf.matmul(x, tf.transpose(Kmat))
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