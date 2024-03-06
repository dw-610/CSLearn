"""
This script will test the integration of the Wasserstein loss into the CSLearn
framework for the classifier model.
"""

# ------------------------------------------------------------------------------
# imports

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np

from cslearn.controllers import ImageLearningController

# ------------------------------------------------------------------------------

def main():
    
    # parameters
    M = np.array(
        [
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
        ]
    ).astype(np.float32)/9.0
    BS = 32
    LD = 2
    EPOCHS = 10
    WP = 1.0
    WL = 1.0
    LOSS = 'categorical_crossentropy'

    # main code
    ctrl = ImageLearningController('classifier', debug=False)
    ctrl.create_data_loaders('mnist', batch_size=BS)
    ctrl.create_learner(latent_dim=LD, output_activation='linear')
    ctrl.summarize_models()
    ctrl.compile_learner(
        loss=LOSS,
        metrics=['accuracy'],
        metric_matrix=M,
        wasserstein_lam=WL,
        wasserstein_p=WP,
        schedule_type='cosine',
        sch_init_lr=0.0001,
        sch_warmup_steps=3*1876,
        sch_warmup_target=0.001,
        sch_decay_steps=7*1876,
    )
    ctrl.train_learner(epochs=EPOCHS)
    ctrl.eval_plot_accuracy_curves()
    ctrl.eval_plot_scattered_features(
        legend = [str(i) for i in range(10)]
    )

    print('\nFinished running - no errors.\n')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------