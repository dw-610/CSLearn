# `cslearn` Package Documentation

This document provides an overview of the `cslearn` package structure and its components.

## Package Structure

The `cslearn` package is organized into the following modules:

### Core Modules

#### `controllers.py`
Contains the `ImageLearningController` class, which serves as the primary interface for users. This controller wraps the functionality of the other modules and provides a simple API for creating, training, and evaluating deep learning models on image data.

**Key Class:**
- `ImageLearningController` - Main controller class for training image-based models

**See Also:** [Detailed usage guide](usage.md) for complete documentation of the `ImageLearningController` API.

#### `arch/models.py`
Contains the model definitions for the various learner types and architectures.

**Encoder Models:**
- `ConvEncoder` - Custom CNN encoder
- `ResNet18Encoder`, `ResNet34Encoder`, `ResNet50Encoder`, `ResNet101Encoder` - ResNet-based encoders

**Decoder Models:**
- `ConvDecoder` - Custom CNN decoder
- `ResNet18Decoder`, `ResNet34Decoder`, `ResNet50Decoder` - ResNet-based decoders

**Learner Models:**
- `Classifier` - Standard image classification model
- `Autoencoder` - Standard autoencoder for image reconstruction
- `VariationalAutoencoder` - Variational autoencoder (VAE)
- `DomainLearnerModel` - Custom model for learning conceptual space representations

#### `arch/layers.py`
Contains custom layer implementations used in the models.

**Custom Layers:**
- `AWGNLayer` - Additive White Gaussian Noise layer
- `EuclideanDistanceLayer` - Computes Euclidean distances to prototypes
- `GaussianSimilarityLayer` - Computes Gaussian similarity from distances
- `SoftGaussSimPredictionLayer` - Soft-assignment prediction layer
- `ConvolutionBlock` - Configurable convolutional block
- `DeconvolutionBlock` - Configurable deconvolutional block
- `HeightWidthSliceLayer` - Spatial dimension extraction layer
- `SmallResNetBlock`, `ResNetBlock` - ResNet residual blocks
- `SmallDeResNetBlock`, `DeResNetBlock` - Deconvolutional ResNet blocks
- `ReparameterizationLayer` - VAE sampling layer

#### `feeders.py`
Contains classes for creating TensorFlow Dataset objects from various data sources.

**Data Feeder Classes:**
- `ClassifierFeederFromArray` - Loads classification data from numpy arrays
- `ClassifierFeederFromMemmap` - Loads classification data from memory-mapped files
- `AutoencoderFeederFromArray` - Loads autoencoder data from numpy arrays
- `AutoencoderFeederFromMemmap` - Loads autoencoder data from memory-mapped files
- `DomainLearnerFeederFromArray` - Loads domain learner data from numpy arrays
- `DomainLearnerFeederFromMemmap` - Loads domain learner data from memory-mapped files

#### `training.py`
Contains custom training logic for models with specialized loss functions.

**Training Classes:**
- `WassersteinClassifierTrainer` - Custom trainer for Wasserstein loss with classifiers
- `WassersteinDomainLearnerTrainer` - Custom trainer for Wasserstein loss with domain learners

**Helper Functions:**
- Gradient computation functions for various loss types
- Prototype update functions for domain learning

#### `visualization.py`
Contains functions and classes for visualizing training results and learned representations.

**Key Classes:**
- `PrototypePlotter2D` - Real-time prototype visualization during training

**Functions:**
- Various plotting functions for loss curves, latent spaces, reconstructions, etc.

#### `callbacks.py`
Contains custom Keras callbacks for enhanced training monitoring.

**Callback Classes:**
- `LearningRateLogger` - Logs learning rate at each epoch
- `ImageLoggerCallback` - Logs images to Comet ML during training

#### `utilities.py`
Contains general-purpose helper functions.

**Functions:**
- `get_unused_name()` - Generates unique filenames to avoid overwrites
- `one_hot_to_ints()` - Converts one-hot encoding to integer labels
- `print_model_summary()` - Prints model summary compatible with custom layers

## Learner Types

The `cslearn` package supports four types of learners:

### 1. Classifier
A standard image classification model that uses an encoder to reduce the input image to a latent vector, which is then passed to a dense layer for classification.

**Use Cases:**
- Image classification tasks
- Feature extraction
- Transfer learning

### 2. Autoencoder
An encoder-decoder architecture that learns to reconstruct input images. Available in both standard and variational forms.

**Use Cases:**
- Dimensionality reduction
- Image denoising
- Feature learning
- Generative modeling (variational)

### 3. Domain Learner
A custom architecture that combines autoencoding with semantic regularization based on prototype-based reasoning. This learner is designed for learning conceptual space representations.

**Use Cases:**
- Semantic communication applications
- Conceptual space learning
- Multi-task learning (reconstruction + property prediction)
- Interpretable representation learning

### 4. Space Learner (Experimental)
An extension of the domain learner that supports learning multiple domains within a conceptual space.

**Use Cases:**
- Multi-domain conceptual space learning
- Complex semantic representation tasks

## Architecture Options

The `cslearn` package supports multiple CNN architectures:

- **`custom_cnn`** - Fully configurable custom CNN architecture where you can specify the number of blocks, filters, kernel sizes, strides, and pooling options
- **`resnet18`** - ResNet-18 architecture
- **`resnet34`** - ResNet-34 architecture
- **`resnet50`** - ResNet-50 architecture
- **`resnet101`** - ResNet-101 architecture (encoder only)

## Supported Datasets

Out of the box, the package supports:
- **MNIST** - Handwritten digit recognition
- **CIFAR-10** - Object classification (10 classes)
- **Local datasets** - Custom datasets provided as memory-mapped numpy arrays

## Installation

See the main [README](../README.md) for installation instructions.

## Getting Started

For a quick introduction to using the package, see the [Quickstart Guide](../README.md#quickstart-guide) in the README.

For detailed documentation on all available options and methods, see the [Usage Guide](usage.md).

For working examples, check out the Jupyter notebooks:
- `demo_classifier.ipynb` - Classifier examples
- `demo_autoencoder.ipynb` - Autoencoder and VAE examples
- `demo_domainLearner.ipynb` - Domain learner examples