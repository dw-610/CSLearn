# Introduction

The code in this repository provides an easy-to-use framework for creating and training various deep learning models in Python with Keras/Tensorflow. Currently geared toward image data, all models are based on convolutional neural networks (CNNs). The `cslearn` package provides a `controllers.py` wrapper module for getting up and running quickly. 

Syntactically, the overall model that is trained is referred to here as the *learner*, and each learner is made up of one or more sub-models. For example, an autoencoder learner is made up of encoder and decoder sub-models. There are several types of learners available:

- Classifier
- Autoencoder
    - Standard and Variational
- Domain Learner

The Domain Learner type is a custom architecture that is based on the autoencoder and was developed for learning conceptual space representations of a dataset. For more details on this, see [[1]](#references) in the [References](#references) section.

This code was originally developed with the goal of learning conceptual spaces for semantic communication. For background on semantic communication, see [[2]](#references). For more details on the theory of conceptual spaces, see [[3,4]](#references). For more information regarding semantic communication with conceptual spaces, see [[5,6]](#references)



# Contents

- [Quickstart Guide](#quickstart-guide)
- [Setting Up the Environment](#setting-up-the-environment)
- [References](#references)



# Quickstart Guide

Included in the `cslearn` package is a module `controllers.py` containing (at present) a single class definition `ImageLearningController` which is a wrapper around the rest of `cslearn` facilitating easy creation, training, and evaluation of deep learning models for image data.

See the [Setting Up the Environment](#setting-up-the-environment) section for ensuring you have all of the necessary dependencies installed. Once the environment is ready, a minimal example implementing the entire workflow to create and train a classifier on the MNIST handwritten digits dataset [[7]](#references) is given below.

```python
from cslearn.controllers import ImageLearningController
ctrl = ImageLearningController(learner_type='classifier')
ctrl.create_data_loaders(dataset='mnist')
ctrl.create_learner(latent_dim=16, architecture='custom_cnn')
ctrl.compile_learner(loss='categorical_crossentropy',metrics=['accuracy'])
ctrl.train_learner(epochs=3)
ctrl.eval_plot_loss_curves()
```

Each step of the workflow can be customized with the arguments provided to the various methods. For more details about the options available, take a look at this [detailed usage guide](docs/usage.md).

Also, see the Jupyter notebook files named `demo_*.ipynb` for more code examples.

# Setting Up the Environment

A simple way to set up the environment with the necessary packages is with an Anaconda
environment. Details for installing the `conda` package manager can be found [here](https://docs.anaconda.com/free/anaconda/install/index.html).

Commands for setting up the environment for both Linux and Windows systems are provided below. Both
methods should result in a GPU-compatible installation of Tensorflow. Note that there are some other
software requirements for using Tensorflow with a GPU, see [here](https://www.tensorflow.org/install/pip#software_requirements) for more details.

The following commands have been tested using `conda` version 23.11.0.

### Linux

The YAML file `env/linux_env.yml` contains instructions for setting up the Anaconda environment. Once Anaconda is installed, open a terminal and navigate to the `env/` directory. From here, run the command:
```
conda env create -f linux_env.yml
```
After the environment has been successfully created, run
```
conda activate cslearn-env
```
to begin working with CSLearn.

### Windows

The YAML file `env/windows_env.yml` contains instructions for setting up the Anaconda environment. Once Anaconda is installed, open the Anaconda prompt and navigate to the `env/` directory. From here, run the command:
```
conda env create -f windows_env.yml
```
After the environment has been successfully created, run
```
conda activate cslearn-env
```
to begin working with CSLearn.

# References

**[1]** D. Wheeler and B. Natarajan, "Autoencoder-Based Domain Learning for Semantic Communication with Conceptual Spaces," 2024. arXiv: [2401.16569 [cs.LG]](https://arxiv.org/abs/2401.16569)  
**[2]** D. Wheeler and B. Natarajan, "Engineering Semantic Communication: A Survey," *IEEE Access*, vol. 11, pp. 13965-13995, 2023. [(link)](https://ieeexplore.ieee.org/document/10038657)  
**[3]** P. G&auml;rdenfors, *Conceptual Spaces: The Geometry of Thought.* Massachusetts Institute of Technology, 2000. [(link)](https://direct.mit.edu/books/book/2532/Conceptual-SpacesThe-Geometry-of-Thought)  
**[4]** P. G&auml;rdenfors, *The Geometry of Meaning: Semantics Based on Conceptual Spaces.* Massachusetts Institute of Technology, 2014. [(link)](https://direct.mit.edu/books/book/4012/The-Geometry-of-MeaningSemantics-Based-on)    
**[5]** D. Wheeler, E. E. Tripp, and B. Natarajan, "Semantic Communication with Conceptual Spaces," *IEEE Communications Letters*, vol. 27, no. 2, pp. 532-535, 2023. [(link)](https://ieeexplore.ieee.org/document/9991159)  
**[6]** D. Wheeler and B. Natarajan, "Knowledge-Driven Semantic Communication Enabled by the Geometry of Meaning," 2023. arXiv: [2306.02917 [eess.SP]](https://arxiv.org/abs/2306.02917)  
**[7]** Y. LeCun, C. Cortes, and C. Burges, "MNIST handwritten digit database," *ATT Labs* [Online]. Available: http://yann.lecun.com/exdb/mnist