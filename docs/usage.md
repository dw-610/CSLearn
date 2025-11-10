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
The loss function to use for training.
Options for 'classifier' are 'categorical_crossentropy' or 'wasserstein'.
Options for 'autoencoder' are 'mse' or 'ssim'.
Options for 'domain_learner' are 'basic' or 'wasserstein'.
Default is 'mse'.
- `optimizer : str (Default = 'adam')`
The optimizer to use for training. Currently, the only option is 'adam'.
- `learning_rate : float (Default = 1e-3)`
The learning rate to use for training.
- `weight_decay : float (Default = None)`
The strength of the weight decay to use for training. If None, no weight decay is used.
- `clipnorm : float (Default = None)`
If set, the gradient of each weight is individually clipped so that its norm is no higher than this value.
- `clipvalue : float (Default = None)`
If set, the gradient of each weight is clipped to be no higher than this value.
- `metrics : list (Default = None)`
A list of metrics to use for training. For example, `['accuracy']` for classifier models.
- `alpha : float (Default = 1.0)`
The weight for the reconstruction loss. Only used if the learner_type is 'domain_learner'.
- `beta : float (Default = 1.0)`
The weight for the classification loss. Only used if the learner_type is 'domain_learner'.
- `lam : float (Default = 0.01)`
The weight for the semantic distance regularization term. Only used if the learner_type is 'domain_learner'.
- `schedule_type : str (Default = None)`
String identifier for the learning rate schedule to use. Options are 'cosine' or None.
- `sch_init_lr : float (Default = 1e-4)`
The initial learning rate for the schedule. Only used if schedule_type is not None.
- `sch_decay_steps : int (Default = 10000)`
The number of steps (batches, not epochs) before decay. Only used if schedule_type is not None.
- `sch_warmup_target : float (Default = None)`
The target learning rate for the warmup phase. Only used if schedule_type is 'cosine'.
- `sch_warmup_steps : int (Default = None)`
The number of steps for the warmup phase. Only used if schedule_type is 'cosine'.
- `metric_matrix : np.ndarray (Default = None)`
The matrix of distances between the classes. Only used for the Wasserstein loss. For the domain_learner, if None, the matrix is dynamically computed from the prototypes learned during training.
- `wasserstein_lam : float (Default = 1.0)`
The balancing parameter for the Wasserstein loss. Only used when loss is 'wasserstein'.
- `wasserstein_p : float (Default = 1.0)`
The exponent for the distance metric in the Wasserstein loss. To get a valid metric, this should be >= 1. Only used when loss is 'wasserstein'.
- `scaled_prior : bool (Default = False)`
Whether to use the scaled prior for the VAE. Only used for the variational autoencoder or variational domain_learner.

[(back to contents)](#contents)

### 4. Training the Learner

After compiling the learner, we can train it. This is done with the `.train_learner()` method:

```python
ctrl.train_learner(epochs=10)
```

This method trains the model(s) using the data loaders that were created in step 1.

#### Required Arguments

None

#### Optional Arguments

- `epochs : int (Default = 5)`
The number of epochs to train for.
- `steps_per_epoch : int (Default = None)`
The number of batches trained on per epoch. If None, the entire training dataset is used.
- `validation_steps : int (Default = None)`
The number of batches validated on per epoch. If None, the entire validation dataset is used.
- `callbacks : list (Default = None)`
A list of `tf.keras.callbacks.Callback` objects to use for training. These will be passed directly to the model.fit method.
- `verbose : int (Default = 1)`
The verbosity level to use for training. 1 is progress bar, 2 is one line per epoch, 0 is silent.
- `proto_update_type : str (Default = 'average')`
The type of prototype update to use. Options are 'average'. Only applies to domain_learner.
- `proto_update_step_size : int (Default = None)`
The number of batches used to update the prototypes. Only applies to domain_learner.
- `mu : float (Default = 0.5)`
The "mixing parameter" for the prototype update. 1.0 just uses the old prototype, 0.0 is full update. Only applies to domain_learner.
- `warmup : int (Default = 0)`
The number of epochs to train without semantic regularization. Only applies to domain_learner.
- `log_experiment : bool (Default = False)`
Whether to log the experiment to Comet ML. Requires that the `comet_info.json` file is properly configured.
- `proto_plot_save_path : str (Default = None)`
The path to save the prototype plots to. Only applies to domain_learner with 2D latent space.
- `proto_plot_colors : list (Default = None)`
A list of colors to use for the prototype plots. Only applies to domain_learner.
- `proto_plot_legend : list (Default = None)`
A list of strings to use for the legend in the prototype plots. Only applies to domain_learner.
- `fixed_prototypes : bool (Default = False)`
Whether to keep the prototypes fixed during training (i.e., not update them). Only applies to domain_learner.

[(back to contents)](#contents)

### 5. Evaluating the Results

After training the learner, there are several methods available for evaluating the results and visualizing what was learned. All evaluation methods begin with the prefix `eval_`. The available methods are described below.

#### `eval_plot_loss_curves()`

Plots the training and/or validation loss curves over the course of training.

**Optional Arguments:**
- `which : str (Default = 'both')`
Which loss curves to plot. Options are 'training', 'validation', or 'both'.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_plot_loss_curves()
```

#### `eval_plot_accuracy_curves()`

Plots the training and/or validation accuracy curves over the course of training. Only applicable if accuracy was included as a metric during compilation.

**Optional Arguments:**
- `which : str (Default = 'both')`
Which accuracy curves to plot. Options are 'training', 'validation', or 'both'.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_plot_accuracy_curves()
```

#### `eval_compare_latent_prior()`

Compares the learned latent space distribution to the prior distribution. Only applicable for variational autoencoder or variational domain_learner.

**Optional Arguments:**
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_compare_latent_prior()
```

#### `eval_plot_scattered_features()`

Plots the learned latent features in 2D or 3D space. This method is useful for visualizing how the encoder has learned to cluster different classes in the latent space. Only works if latent_dim is 2 or 3.

**Optional Arguments:**
- `which : str (Default = 'validation')`
Which features to plot. Options are 'training' or 'validation'.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.
- `colors : list (Default = None)`
A list of colors to use for the different classes. If None, colors are automatically chosen.
- `legend : list (Default = None)`
A list of strings to use for the legend. If None, no legend is shown.

**Example:**
```python
ctrl.eval_plot_scattered_features(colors=['red', 'blue'], legend=['Class 0', 'Class 1'])
```

#### `eval_show_decoded_protos()`

Visualizes the decoded prototype representations. Only applicable for domain_learner.

**Optional Arguments:**
- `legend : list (Default = None)`
A list of strings to use for the legend.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_show_decoded_protos()
```

#### `eval_plot_scattered_protos()`

Plots the learned prototypes in the latent space. Only applicable for domain_learner with 2D or 3D latent space.

**Optional Arguments:**
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.
- `colors : list (Default = None)`
A list of colors to use for the different prototypes.
- `legend : list (Default = None)`
A list of strings to use for the legend.

**Example:**
```python
ctrl.eval_plot_scattered_protos()
```

#### `eval_compare_true_and_generated()`

Compares the original input images with their reconstructions from the autoencoder or domain_learner. This method displays a grid of images showing the input and output side-by-side.

**Optional Arguments:**
- `number_of_samples : int (Default = 10)`
The number of samples to display.
- `which : str (Default = 'validation')`
Which dataset to use. Options are 'training' or 'validation'.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_compare_true_and_generated(number_of_samples=5)
```

#### `eval_plot_similarity_heatmap()`

Plots a heatmap showing the similarity between different classes in the learned conceptual space. Only applicable for domain_learner.

**Optional Arguments:**
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_plot_similarity_heatmap()
```

#### `eval_visualize_dimension()`

Visualizes how a single dimension of the latent space affects the reconstructed output. This method generates a series of images by varying one dimension while keeping others fixed.

**Required Arguments:**
- `dimension : int`
The index of the dimension to visualize (0-indexed).

**Optional Arguments:**
- `number_of_steps : int (Default = 10)`
The number of steps to take along the dimension.
- `range_min : float (Default = -3.0)`
The minimum value for the dimension.
- `range_max : float (Default = 3.0)`
The maximum value for the dimension.
- `fixed_values : np.ndarray (Default = None)`
The fixed values to use for the other dimensions. If None, zeros are used.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_visualize_dimension(dimension=0, number_of_steps=15)
```

#### `eval_visualize_all_dimensions()`

Visualizes how all dimensions of the latent space affect the reconstructed output by calling `eval_visualize_dimension()` for each dimension.

**Optional Arguments:**
- `number_of_steps : int (Default = 10)`
The number of steps to take along each dimension.
- `range_min : float (Default = -3.0)`
The minimum value for each dimension.
- `range_max : float (Default = 3.0)`
The maximum value for each dimension.
- `fixed_values : np.ndarray (Default = None)`
The fixed values to use for the other dimensions. If None, zeros are used.
- `show : bool (Default = True)`
Whether to show the plots.
- `save_path : str (Default = None)`
The base path to save the plots to. Each dimension will be saved with an appended index.
- `block : bool (Default = True)`
Whether to block the execution until the plot windows are closed.

**Example:**
```python
ctrl.eval_visualize_all_dimensions()
```

#### `eval_plot_similarity_histograms()`

Plots histograms of the similarity values for each property in the domain_learner. Only applicable for domain_learner.

**Optional Arguments:**
- `which : str (Default = 'validation')`
Which dataset to use. Options are 'training' or 'validation'.
- `show : bool (Default = True)`
Whether to show the plot.
- `save_path : str (Default = None)`
The path to save the plot to. If None, the plot is not saved.
- `block : bool (Default = True)`
Whether to block the execution until the plot window is closed.

**Example:**
```python
ctrl.eval_plot_similarity_histograms()
```

[(back to contents)](#contents)

# References

**[1]** D. Wheeler and B. Natarajan, "Autoencoder-Based Domain Learning for Semantic Communication with Conceptual Spaces," 2024. arXiv: [2401.16569 [cs.LG]](https://arxiv.org/abs/2401.16569)  

[(back to contents)](#contents)