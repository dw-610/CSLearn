def main():
    # --------------------------------------------------------------------------
    # imports

    import numpy as np
    import tensorflow as tf

    from cslearn.controllers import ImageLearningController

    # --------------------------------------------------------------------------
    # constants

    BATCH_SIZE = 64
    LATENT_DIM = 2
    ARCH = 'custom_cnn'
    EPOCHS = 10
    LEARNING_RATE = 1e-3

    # --------------------------------------------------------------------------
    # main

    ctrl = ImageLearningController(learner_type='classifier')
    
    ctrl.create_data_loaders('mnist', batch_size=BATCH_SIZE)
    ctrl.create_learner(LATENT_DIM, ARCH, global_pool_type='max')
    ctrl.compile_learner(
        'categorical_crossentropy', 
        'adam', 
        LEARNING_RATE,
        metrics=['accuracy']
    )
    ctrl.summarize_models()
    ctrl.train_learner(EPOCHS)

    legend = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    ctrl.eval_plot_loss_curves()
    ctrl.eval_plot_accuracy_curves()
    ctrl.eval_plot_scattered_features(legend=legend)

    # --------------------------------------------------------------------------

if __name__ == '__main__':
    main()