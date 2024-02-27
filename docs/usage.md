# Usage

The `cs_learn` package provides easy-to-use wrappers (termed *controllers*) through the `controllers.py` module that abstract away much of the workflow and allow for simple custimization of the process. Each controller is dedicated toward a specific data modality. Currently available controllers include:
- ImageLearningController

Each controller breaks the workflow down into four primary steps:
1. Data loader creation
2. Learner creation
3. Learner compilation
4. Learner training
5. Evaluation

Each step has it's own associated method(s) that perform the needed functionality and allow for customization. This document provides details on each step of the process.

# Contents

- [`ImageLearningController`](#imagelearningcontroller)
    - [1. Setting Up the Data Loaders](#1-setting-up-the-data-loaders)
    - [2. Creating the Learner](#2-creating-the-learner)
    - [3. Compiling the Learner](#3-compiling-the-learner)
    - [4. Training the Learner](#4-training-the-learner)
    - [5. Evaluating the Results](#5-evaluating-the-results)

# `ImageLearningController`

The `ImageLearningController` class is designed for learning on images which take the form of 3-dimensional arrays/tensors (height x width x channels).

To instantiate an object of this class, use

```python
ctrl = ImageLearningController(learner_type='learner_type')
```

The `learner_type` argument is mandatory and specifies which learner you intend to train, options include: 
- `'classifier'`: This learner uses an encoder model to reduce the dimension of the input image to a single vector, which is then passed to a single dense layer with dimension equal to the number of classes.
- `'autoencoder'`: This learner uses an encoder model to reduce the dimension of the the input image to a single vector. This vector is passed to a decoder model, which attempts to reconstruct the input image from the encoded vector representation.
- `'domain_learner'`: This learner uses the same architecture as the autoencoder, with an added "semantic regularization" module. Please see [[1]](#references) for details on the architecture of the domain learner.
- `'space_learner'`: Experimental learner which is very similar to the `'domain_learner'`, but extends this framework to learning a conceptual space with multiple domains.

[(back to contents)](#contents)

### 1. Setting Up the Data Loaders

The first step in the workflow is to set up the data loaders that will be used to handle the data being passed to the learner during training. The `.create_data_loaders()` method is used to accomplish this:

```python
ctrl.create_data_loaders(dataset='dataset')
```

This method creates some `tf.Data.Dataset` objects that are subsequently used during training and evaluation of the learner.

#### Required Arguments

- `dataset : str`  
The name of the dataset to load in. Options are `'mnist'`, `'cifar10'`, or `'local'`.  
MNIST and CIFAR-10 datasets are provided out of the box from Tensorflow, and can be used without any other changes.  
The `'local'` option can be used to specify that a local dataset is present and will be used for training. Note that the data must meet some requirements to use this option:
    - It must be saved as four separate memory-mapped numpy arrays (training data, training labels, validation data, validation labels). See [here](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) for saving memory-mapped arrays.
    - The paths to and shapes of the data arrays must be specified with the `paths_dict` and `shapes_dict` arguments (see details below).
    - Each array must be saved as type `np.float32` to ensure correct loading.

#### Optional Arguments

- `batch_size : int (Default = 32)`  
The minibatch size used during training.
- `buffer_size : int (Default = 10000)`  
The buffer size used by the tf.Data.Dataset object used for random minibatch sampling during training.
- `paths_dict : dict (Default = None)`  
A dictionary containing the paths to the data and labels arrays, if
using 'local' data. For 'classifier' and 'domain_learner' models,
the dictionary should be of the form:  
{  
&nbsp;&nbsp;&nbsp;&nbsp;'train_data_path': 'path/to/train_data.npy',  
&nbsp;&nbsp;&nbsp;&nbsp;'train_labels_path': 'path/to/train_labels.npy',  
&nbsp;&nbsp;&nbsp;&nbsp;'valid_data_path': 'path/to/train_data.npy',  
&nbsp;&nbsp;&nbsp;&nbsp;'valid_labels_path': 'path/to/train_labels.npy'  
}  
For 'autoencoder' models, the dictionary should be of the form:  
{  
&nbsp;&nbsp;&nbsp;&nbsp;'train_data_path': 'path/to/train_data.npy',  
&nbsp;&nbsp;&nbsp;&nbsp;'valid_data_path': 'path/to/train_data.npy'  
}
- `shapes_dict : dict (Default = None)`  
A dictionary containing the shapes of the data and labels arrays, if
using 'local' data.
For 'classifier' and 'domain_learner' models, the dictionary should
be of the form:  
{  
&nbsp;&nbsp;&nbsp;&nbsp;'train_data_shape': (train_samples, height, width, channels),  
&nbsp;&nbsp;&nbsp;&nbsp;'train_labels_shape': (train_samples, number_of_properties),  
&nbsp;&nbsp;&nbsp;&nbsp;'valid_data_shape': (valid_samples, height, width, channels),  
&nbsp;&nbsp;&nbsp;&nbsp;'valid_labels_shape': (valid_samples, number_of_properties)  
}  
For 'autoencoder' models, the dictionary should be of the form:  
{  
&nbsp;&nbsp;&nbsp;&nbsp;'train_data_shape': (train_samples, height, width, channels),  
&nbsp;&nbsp;&nbsp;&nbsp;'valid_data_shape': (valid_samples, height, width, channels)  
}

[(back to contents)](#contents)

### 2. Creating the Learner

After setting up the data loaders, we can instantiate the learner. This is done with the `.create_learner()` method:

```python
ctrl.create_learner(latent_dim=latent_dim)
```

This method creates some `tf.keras.models.Model` objects that define the networks that are trained throughout the learning process.

#### Required Arguments

- `latent_dim : int`  
The dimension of the latent space learned. In the 'classifier' learner type, this is the dimension of the second-to-last dense layer (before the classification layer). In all of the models, this is the dimension of the vector that the input is 'compressed' to, which serves as the input to the decoder model.

#### Optional Arguments

- `architecture : str (Default = 'custom_cnn')`  
The architecture of the convolutional models. Options are 'custom_cnn' (for which the details of the architecture of specified in additional arguments), 'resnet18', 'resnet34' or 'resnet50'.
- `autoencoder_type : str (Default = 'standard')`  
Applicable to all learner types except the classifier. Options are 'standard' or 'variational'.
- `number_of_blocks : int (Default = 4)`  
Only used if architecture='custom_cnn'. This argument specifies the number of convolutional blocks in the encoder (and decoder, if the learner is not classifier).
- `filters : int or list (Default = [16,16,32,32])`  
Only used if architecture='custom_cnn'. Specifies the number of filters to use in each convolutional block. If an `int` is given, each block has that number of filters.
- `kernel_sizes : int or list (Default = [7,5,3,3])`  
Only used if architecture='custom_cnn'. Specifies the size of the kernal matrix to use in each convolutional block. If an `int` is given, each block uses that kernel size.
- `strides : int or list (Default = [2,1,2,1])`  
Only used if architecture='custom_cnn'. Specifies the stride length used in each convolutional block. If an `int` is given, each block uses that stride length.
- `use_maxpool : bool or list (Default = False)`  
Only used if architecture='custom_cnn'. Specifies whether 2x2 max pooling should be applied at the end of each convolutional block. If a single `bool` is given, each block uses max pooling according to the specified value.
- `hidden_activation : str (Default = 'relu')`  
The activation function to use in the hidden layers of the models. Options are 'relu', 'selu', 'gelu' or 'linear' (no activation).
- `latent_activation : str (Default = 'linear')`  
The activation function to use in the final layer outputting the latent representation. Options are 'relu', 'selu', 'gelu', or 'linear' (no activation).
- `output_activation : str (Default = 'linear')`  
The activation function to use in the final layer of the learner. Options are 'linear' or 'sigmoid'.
- `global_pool_type : str (Default = 'avg')`  
The type of global pooling to use when converting the convolutional features of the encoder to the latent representation vector. Options are 'avg', 'max', or None (where the features are just flattened instead of pooled).
- `use_awgn : bool (Default = False)`  
Whether to add white Gaussian noise to the inputs during training.
- `awgn_variance : float (Default = 0.1)`  
If use_awgn=True, this parameter specifies the variance of the AWGN added to the input data samples.
- `distance : str (Default = 'euclidean')`  
Only used if the learner_type is domain_learner OR space_learner. Specifies the distance metric assumed for the conceptual space. Currently, the only option is 'euclidean'.
- `similarity : str (Default = 'gaussian')`  
Only used if the learner_type is domain_learner OR space_learner. Specifies the similarity function assumed for the conceptual space. Currently, the only option is 'gaussian'.
- `similarity_c : float (Default = 1.0)`  
Only used if the learner_type is domain_learner OR space_learner. Specifies the parameter c used in the similarity function.
- `initial_protos : np.ndarray (Default = None)`  
Only used if the learner_type is domain_learner OR space_learner. If a `np.ndarray` is given, these are the initial prototypes used in the learning process. If None is given, prototypes are randomly initialized.
- `domain_mask : np.ndarray (Default = None)`  
Only used if the learner_type is space_learner. If a `np.ndarray` is given, the mask matrix is fixed at this value. If None is given, then the matrix is randomly initialized and learned throughout the training process.
- `number_of_potential_domains : int (Default = 2)`  
Only used if the learner_type is space_learner. Specifies the number of potential domains to be learned in the conceptual space.

[(back to contents)](#contents)

### 3. Compiling the Learner

Once the learner has been created, it needs to be compiled. This is done with

```python
ctrl.compile_learner()
```

This method compiles the models associated with the learner, and sets loss function- and optimizer-related parameters.

#### Required Arguments

None

#### Optional Arguments

- `loss : str (Default = 'mse')`  
Only used if the learner_type is classifier OR autoencoder
- `optimizer : str (Default = 'adam')`
- `learning_rate : float (Default = 1e-3)`
- `metrics : list (Default = None)`
- `alpha : float (Default = 1.0)`
- `beta : float (Default = 1.0)`
- `lam : float (Default = 0.01)`
- `gamnma : float (Default = 1.0)`

[(back to contents)](#contents)

### 4. Training the Learner

#### Required Arguments

#### Optional Arguments

[(back to contents)](#contents)

### 5. Evaluating the Results


[(back to contents)](#contents)

# References

**[1]** D. Wheeler and B. Natarajan, "Autoencoder-Based Domain Learning for Semantic Communication with Conceptual Spaces," 2024. arXiv: [2401.16569 [cs.LG]](https://arxiv.org/abs/2401.16569)  

[(back to contents)](#contents)