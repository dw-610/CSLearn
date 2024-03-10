# add-wasserstein-vae branch notes

This file contains notes for work on this branch.

Goal:

- Add wasserstein loss option when training the VAE

## 03/10/24

Central idea is to use the Wasserstein loss to train a VAE.

To compute Wasserstein loss, we need three things:

1. Input distribution (discrete) or measure/density (continuous)
2. Target distribution (discrete) or measure/density (continuous)
3. Ground metric over the input and target space

In the case of a classifer, we have

1. The input distribution is the predicted distribution over the classes
2. The target "distribution" is 1 for the true class and 0 elsewhere
    - I.e., it is deterministic
3. The ground metric is a pairwise distance defined between each class

In the case of the VAE, the input/target space is essentially $\mathbb{R}^{H \times X \times C}$, and thus an image can be considered as a vector $\textbf{x} \in \mathbb{R}^{H \times X \times C}$. Then we need:

- *Target distribution.* We can take this to be a Gaussian with $\boldsymbol{\mu} = \textbf{x}$. Let the covariance matrix $\boldsymbol{\Sigma}$ be diagonal, i.e., assume independent pixels (this is obviously not a good assumption - but we can start here). Let the vector representing the diagonal of this matrix be $\boldsymbol{\sigma}$.  
Analogous to the discrete case, we can make the target distribution "deterministic" by letting $\boldsymbol{\sigma} = \textbf{0}$.
  - This might cause issues? If so we can take the variances to be very small values
- *Input distribution.* Assume that the outputs are also multivariate independent Gaussians: $$ \hat{\textbf{x}} \mid \textbf{z} \sim \mathcal{N}(\hat{\boldsymbol{\mu}}, \text{diag}(\boldsymbol{\hat{\sigma}}^2)) $$  
where $\hat{\boldsymbol{\mu}}$ and $\hat{\boldsymbol{\sigma}}$ are outputs of the decoder. Then both the input and target measures are "diagonal" multivariate Gaussians, with different mean and variance parameters.
- *Ground metric*. We can just take the ground metric to be $$ d(\textbf{x}, \textbf{y}) = \Vert \textbf{x} - \textbf{y} \Vert $$  
If we let $p = 2$, we then get the simple squared divergence $$ d(\textbf{x}, \textbf{y})^2 = \Vert \textbf{x} - \textbf{y} \Vert^2 $$

The 2-Wasserstein distance between two marginal distributions is defined as $$ W_2^2(\alpha, \beta) \stackrel{\Delta}{=} \min_{\pi \in \Pi(\alpha, \beta)} \int_{\mathbb{R}^d} \Vert x - y \Vert^2 d\pi(x,y) $$

Let $\textbf{x} \sim \boldsymbol{\alpha} = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\hat{\textbf{x}} \sim \boldsymbol{\beta} = \mathcal{N}(\hat{\boldsymbol{\mu}}, \hat{\boldsymbol{\Sigma}})$, where $\boldsymbol{\Sigma} = \text{diag}(\boldsymbol{\sigma}^2)$ and $\hat{\boldsymbol{\Sigma}} = \text{diag}(\boldsymbol{\hat{\sigma}}^2)$. Then from [1] we have $$ W_2^2(\boldsymbol{\alpha}, \boldsymbol{\beta}) = \Vert \boldsymbol{\mu} - \hat{\boldsymbol{\mu}} \Vert^2 - \mathcal{B}^2(\boldsymbol{\Sigma}, \hat{\boldsymbol{\Sigma}}) $$
where $\mathcal{B}$ is the *Bures* distance between positive matrices: $$ \mathcal{B}^2(\boldsymbol{\Sigma}, \hat{\boldsymbol{\Sigma}}) \stackrel{\Delta}{=} \text{Tr}(\boldsymbol{\Sigma}) + \text{Tr}(\hat{\boldsymbol{\Sigma}}) - 2\text{Tr}(\boldsymbol{\Sigma}^\frac{1}{2}\hat{\boldsymbol{\Sigma}}\boldsymbol{\Sigma}^\frac{1}{2})^\frac{1}{2} $$

Since we are dealing with diagonal matrices, this simplifies significantly. First, the trace is just $$ \text{Tr}(\boldsymbol{\Sigma}) = \sum_i \sigma_i^2 $$
and the square root of the matrix is the square root of each element. So we have $$ \text{Tr}(\boldsymbol{\Sigma}^\frac{1}{2}\hat{\boldsymbol{\Sigma}}\boldsymbol{\Sigma}^\frac{1}{2})^\frac{1}{2} = \sqrt{\sum_i \sigma_i^2 \hat{\sigma}_i^2} $$
and the Bures distance is then just given by $$ \mathcal{B}^2(\boldsymbol{\Sigma}, \hat{\boldsymbol{\Sigma}}) = \sum_i \sigma_i^2 + \sum_i \hat{\sigma}_i^2 - 2\sqrt{\sum_i \sigma_i^2 \hat{\sigma}_i^2} $$
Then the 2-Wasserstein distance is just $$ \boxed{W_2^2(\boldsymbol{\alpha}, \boldsymbol{\beta}) = \Vert \boldsymbol{\mu} - \hat{\boldsymbol{\mu}} \Vert^2 - \Vert \boldsymbol{\sigma} \Vert^2 - \Vert \hat{\boldsymbol{\sigma}} \Vert^2 + 2 \Vert \boldsymbol{\sigma} \circ \hat{\boldsymbol{\sigma}} \Vert} $$

### Notes/Comments

- The above metric does not include entropic regularization
- If we assume that the target variances are approximately zero, the metric simplifies even further
  - This holds even in the non-diagonal case; left with just the mean and esimated variance-only terms
    - This is very close to the VAE loss term based on the ELBO
- Might need to look at more complicated problem set up to get any significant differences from existing approach

### Reference

[1] Janati et al, "Entropic Optimal Tranport between Unbalanced Gaussian Measures has a Closed Form", 2020.
