# add-wasserstein branch notes

This file contains notes taken for work on this branch.

Goal progression:

- Add Wasserstein loss as an option for training the classifier models.
- Add this loss as an option when training the classifier module of the domain learner.
- Stretch: use this loss to guide the distribution-based VAE learning as well?

## 03/04/24

To do:

- Add Wasserstein as a loss option when training the basic classifier network
  - Figure out how the loss is calculated (look to Frogner et al.)
  - Implement the loss in a custom function
  - Feed this function to the keras workflow to use it for training

Notes:

- First attempt: try to just impliment the iterative loss computation in a custom loss function that is passed to model.compile
  - This may have cause issues with the gradients? Not sure
  - Not applicable (directly) - Frogner paper only gives algorithm for computing the *gradient*, not the loss itself
- Will continue with trying to impliment this gradient methods
  - Essentially, compute the gradient $\partial W_p^p / \partial h(x)$ as in the paper
  - Then use this to back prop over all the other parameters, using the automatic gradients computed for $\partial h(x)/ \partial \theta$
- It works!
  - Implemented the MNIST experiment from the Frogner paper, with numerical distance as the grounding metric
  - Trained a classifier using the custom gradient method mentioned above
  - The resulting feature maps smoothly transitions from 0 to 1 to ... to 9
- *Challenges*
  - The implementation I am using right now requires a custom training loop, i.e. it is not using model.fit()
    - Thus, it is not readily compatible with the current CSLearn setup
  - The magnitude of the distances in $\textbf{M}$ affects the stability of the training, as well as the magnitude of $\lambda$
    - Large $\lambda$ seemed to cause instability, as well as large elements of $\textbf{M}$
      - Can fix this by scaling the elements of $\textbf{M}$

## 03/06/24

To do:

- Integrate Wasserstein loss as an option when training classifiers in CSLearn
  - If Wasserstein is specified, use a custom-defined training loop, else use model.fit() (can retrofit later so that all training methods are consistent, with the custom training loop)
  - Pass training-specified parameters to the compile method (Metric matrix, required for Wasserstein, and the lambda balancing parameter)

Notes:

- Integrated Wassertain loss-based training into the classifier model of CSLearn
  - specify loss as 'wasserstein' when calling compile_learner
  - Pass the metric matrix and (optionally) $\lambda$ and $p$ to compile_learner
  - Automatically tracks the accuracy and saves it to the .history attribute
  - As no loss is computed, loss is not tracked
  - Can use the typical eval methods for the classifier (not eval_plot_loss)

## 03/07/24

To do:

- Integrate Wasserstein loss as an option when training the domain learner
  - Idea is that the loss function now looks like this: $$ \mathcal{L} = \alpha \ell_r + \beta \ell_W $$ where $$ \ell_W = W_p^p - \lambda H $$
  - Note that the Frogner paper uses $1/\lambda$ for the regularization - to be consistent with other domain learner approaches, we want to use just $\lambda$ as a greater value should *increase* regularization/smoothing
  - Should be able to create a custom training loop where updates are done with something like

    ```python
    r_grads = tape.gradient(reconstructed, model.trainable_variables, output_gradients=dlr_dx)
    w_grads = tape.graident(predictions, model.trainable_variables, output_gradients=dlw_dx)
    grads = alpha*r_grads + beta*w_grads
    ```

- Do some testing with the new framework

Notes:

- Added wasserstein option to domain learner framework
  - Right now, not updating M from the prototypes
    - Need to do this when M isn't provided
  - Need to clean up the code a bit
    - Figure out the best way to handle all of the training functions
      - Standalone functions as they are now?
      - Collected in a "trainer" object?
- Preliminary tests (MNIST) seem to confirm that it is working correctly
  - With $\alpha = 0.01$ and $\beta = 100.0$, the feature space seems "mixed" between the Wasserstein-only and the autoencoder-only
  - With $\alpha = 0.0$, it learns the "semi-circle" shape similar to first tests for Wasserstein classifier 
  - With $\beta = 0.0$, it learns the more mixed up representations typical from the autoencoder
- For now, have just kept the regularization parameter as $1/\lambda$

## 03/08/24

To do:

- Clean up code
  - Refactor the custom training loops
    - Create a "trainer" object
      - methods for gradient computations
      - train_step and test_step methods
      - "fit" method for performing the training loop
