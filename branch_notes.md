# add-wasserstein branch notes

This file contains notes taken for work on this branch.

Goal progression:
- Add Wasserstein loss as an option for training the classifier models.
- Add this loss as an option when training the classifier module of the domain learner.
- Stretch: use this loss to guide the distribution-based VAE learning as well?

### 03/04/24
#### To do:
- Add Wasserstein as a loss option when training the basic classifier network
    - Figure out how the loss is calculated (look to Frogner et al.)
    - Implement the loss in a custom function
    - Feed this function to the keras workflow to use it for training
#### Notes:
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


