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
    M: np.ndarray = np.array(
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

    EPOCHS: int = 1000
    BATCH_SIZE: int = 64
    STEPS_PER_EPOCH: int = 200
    VALIDATION_STEPS: int = None
    P_UPDATE_STEP_SIZE: int = 200

    LATENT_DIM: int = 2

    LOSS: str = 'wasserstein'
    WASS_P: float = 1.0
    WASS_LAM: float = 1.0

    ALPHA: float = 1.0
    BETA: float = 100.0
    WARM: int = 0

    # main code
    ctrl = ImageLearningController('domain_learner', debug=False)
    ctrl.create_data_loaders('mnist', batch_size=BATCH_SIZE)
    ctrl.create_learner(latent_dim=LATENT_DIM, output_activation='linear')
    ctrl.summarize_models()
    ctrl.compile_learner(
        loss=LOSS,
        metrics=['accuracy'],
        metric_matrix=M,
        alpha = ALPHA,
        beta = BETA,
        lam = 0.0,
        wasserstein_lam=WASS_LAM,
        wasserstein_p=WASS_P,
        learning_rate=0.001,
        schedule_type='cosine',
        sch_init_lr=0.0001,
        sch_warmup_steps=5*STEPS_PER_EPOCH,
        sch_warmup_target=0.001,
        sch_decay_steps=45*STEPS_PER_EPOCH,
    )
    ctrl.train_learner(
        epochs=EPOCHS, 
        warmup=WARM,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        proto_update_step_size=P_UPDATE_STEP_SIZE,
        mu=0.75,
    )
    legend = [str(i) for i in range(10)]
    ctrl.eval_plot_accuracy_curves()
    ctrl.eval_plot_scattered_features(legend=legend)
    ctrl.eval_plot_scattered_protos(legend=legend)
    ctrl.eval_plot_similarity_heatmap(legend=legend)
    # ctrl.eval_visualize_all_dimensions()
    ctrl.eval_compare_true_and_generated()

    # print(ctrl.prototypes)

    print('\nFinished running - no errors.\n')

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    main()

# ------------------------------------------------------------------------------