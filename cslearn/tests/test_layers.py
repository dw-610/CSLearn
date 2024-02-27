"""
This scripts contains unit tests for the layers.py module.
"""

# ------------------------------------------------------------------------------
# imports

# silence tensorflow info and warnings for the tests
print("\nSilencing Tensorflow info, warnings, and errors...\n")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

import unittest

from cslearn.arch import layers as cs_layers

# ------------------------------------------------------------------------------

class TestAWGNLayer(tf.test.TestCase):

    def setUp(self):
        self.variance = 1.0
        self.layer = cs_layers.AWGNLayer(self.variance)
        self.input_shape = (1,10)
        self.input_data = tf.random.normal(self.input_shape)

    def test_noise_addition_during_training(self):
        """Test if noise is added during training."""
        output = self.layer(self.input_data, training=True)
        self.assertNotEqual(
            tf.reduce_mean(self.input_data), 
            tf.reduce_mean(output)
        )

    def test_no_noise_during_inference(self):
        """Test if no noise is added during inference."""
        output = self.layer(self.input_data, training=False)
        self.assertAllEqual(self.input_data, output)

    def test_output_shape(self):
        """Test that the output shape is correct."""
        output = self.layer(self.input_data, training=True)
        self.assertEqual(output.shape, self.input_shape)

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        output_shape = self.layer.compute_output_shape(self.input_shape)
        self.assertEqual(output_shape, self.input_shape)

# ------------------------------------------------------------------------------
    
class TestEuclideanDistanceLayer(tf.test.TestCase):

    def setUp(self):
        self.prototypes = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
        self.layer = cs_layers.EuclideanDistanceLayer(self.prototypes)

    def test_distance_calculation(self):
        """Test that the distance calculation is correct."""
        input_data = tf.constant([[1.0, 3.0]])
        distances = self.layer(input_data)
        expected_distances = tf.constant([[1.0, np.sqrt(5.0)]])
        self.assertAllClose(distances, expected_distances)

    def test_output_shape(self):
        """Test that the output shape is correct."""
        batch_size = 10
        input_shape = (batch_size, 2)
        input_data = tf.random.uniform(input_shape)
        output = self.layer(input_data)
        self.assertEqual(output.shape, (batch_size, self.prototypes.shape[0]))

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        input_shape = (10, 2)
        output_shape = self.layer.compute_output_shape(input_shape)
        self.assertEqual(
            output_shape, 
            (input_shape[0], 
            self.prototypes.shape[0])
        )

# ------------------------------------------------------------------------------
        
class TestGaussianSimilarityLayer(tf.test.TestCase):

    def setUp(self):
        self.distances = tf.constant([[0.1, 1.0, 2.0]])
        sim_c = 1.0
        self.layer = cs_layers.GaussianSimilarityLayer(sim_c)

    def test_similarity_calculation(self):
        """Test that the similarity calculation is correct."""
        similarities = self.layer(self.distances)
        expected_similarities = tf.exp(-1.0 * tf.square(self.distances))
        self.assertAllClose(similarities, expected_similarities)

    def test_output_shape(self):
        """Test that the output shape is correct."""
        batch_size = 10
        input_data = tf.random.uniform((batch_size, 3))
        output = self.layer(input_data)
        self.assertShapeEqual(input_data, output)

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        input_shape = (10, 3)
        output_shape = self.layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)
    
# ------------------------------------------------------------------------------
        
class TestSoftGaussSimPredictionLayer(tf.test.TestCase):

    def setUp(self):
        self.distances = tf.constant([[10.0, 1.0, 0.5]])
        sim_c = 1.0
        self.layer = cs_layers.SoftGaussSimPredictionLayer(sim_c)

    def test_prediction_correctness(self):
        """Test that the prediction is correct."""
        prediction = self.layer(self.distances)
        self.assertAllLessEqual(prediction[0,:2], prediction[0,2]) 

    def test_prediction_form(self):
        """Test that the prediction vectors sum to 1."""
        inputs = tf.random.uniform((10, 3))
        prediction = self.layer(inputs)
        expected_sums = tf.ones((10,))
        self.assertAllClose(tf.reduce_sum(prediction, axis=1), expected_sums)

    def test_output_shape(self):
        """Test that the output shape is correct."""
        batch_size = 10
        input_data = tf.random.uniform((batch_size, 3))
        output = self.layer(input_data)
        self.assertShapeEqual(input_data, output)

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        input_shape = (10, 3)
        output_shape = self.layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

# ------------------------------------------------------------------------------
        
class TestConvolutionBlock(tf.test.TestCase):

    def setUp(self):
        pass

    def test_base_convolution(self):
        """Test that the convolution operation is correct."""
        layer = cs_layers.ConvolutionBlock(filters=8)
        input_shape = (1, 4, 4, 1)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 4, 4, 8))

    def test_stride_effect(self):
        """Test that the stride parameter has the correct effect."""
        layer = cs_layers.ConvolutionBlock(filters=8, strides=2)
        input_shape = (1, 4, 4, 1)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 2, 2, 8))

    def test_maxpool_effect(self):
        """Test that the maxpool parameter has the correct effect."""
        layer = cs_layers.ConvolutionBlock(filters=8, maxpool=True)
        input_shape = (1, 4, 4, 1)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 2, 2, 8))

    def test_batch_normalization_effect(self):
        """Test that batch normalization has the correct effect."""
        layer = cs_layers.ConvolutionBlock(filters=4, activation=None)
        input_shape = (10000, 4, 4, 1)
        input_data = tf.random.uniform(input_shape)
        output = layer(input_data, training=True)
        means = tf.reduce_mean(output, axis=(0,1,2))
        stds = tf.math.reduce_std(output, axis=(0,1,2))
        self.assertAllClose(means, tf.zeros_like(means), atol=3e-1)
        self.assertAllClose(stds, tf.ones_like(stds), atol=3e-1)
        
    def test_activation_effect(self):
        """Test that the activation parameter has the correct effect."""
        layer = cs_layers.ConvolutionBlock(filters=8, activation='relu')
        input_shape = (1, 4, 4, 1)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertAllGreaterEqual(output, 0.0)

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.ConvolutionBlock(filters=8)
        input_shape = (1, 4, 4, 1)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (input_shape[0], 4, 4, 8))

# ------------------------------------------------------------------------------
    
class TestDeconvolutionBlock(tf.test.TestCase):
    
    def setUp(self):
        pass

    def test_base_deconvolution(self):
        """Test that the deconvolution operation is correct."""
        layer = cs_layers.DeconvolutionBlock(filters=4)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 4, 4, 4))

    def test_stride_effect(self):
        """Test that the stride parameter has the correct effect."""
        layer = cs_layers.DeconvolutionBlock(filters=4, strides=2)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 8, 8, 4))

    def test_unpool_effect(self):
        """Test that the unpool parameter has the correct effect."""
        layer = cs_layers.DeconvolutionBlock(filters=4, unpool=True)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 8, 8, 4))

    def test_batch_normalization_effect(self):
        """Test that batch normalization has the correct effect."""
        layer = cs_layers.DeconvolutionBlock(filters=4, activation=None)
        input_shape = (10000, 4, 4, 8)
        input_data = tf.random.uniform(input_shape)
        output = layer(input_data, training=True)
        means = tf.reduce_mean(output, axis=(0,1,2))
        stds = tf.math.reduce_std(output, axis=(0,1,2))
        self.assertAllClose(means, tf.zeros_like(means), atol=3e-1)
        self.assertAllClose(stds, tf.ones_like(stds), atol=3e-1)

    def test_activation_effect(self):
        """Test that the activation parameter has the correct effect."""
        layer = cs_layers.DeconvolutionBlock(filters=4, activation='relu')
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertAllGreaterEqual(output, 0.0)

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.DeconvolutionBlock(filters=4)
        input_shape = (1, 4, 4, 8)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (input_shape[0], 4, 4, 4))

# ------------------------------------------------------------------------------
            
class TestHeightWidthSliceLayer(tf.test.TestCase):
    
    def setUp(self):
        pass

    def test_output_shape(self):
        """Test that the output shape is correct."""
        layer = cs_layers.HeightWidthSliceLayer(1)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 3, 3, 8))

    def test_output_correctness(self):
        """Test that the output is correct."""
        layer = cs_layers.HeightWidthSliceLayer(1)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        expected_output = input_data[:,:3,:3,:]
        self.assertAllEqual(output, expected_output)

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.HeightWidthSliceLayer(1)
        input_shape = (1, 4, 4, 8)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (input_shape[0], 3, 3, 8))

# ------------------------------------------------------------------------------
        
class TestSmallResNetBlock(tf.test.TestCase):
        
    def setUp(self):
        pass

    def test_base_block(self):
        """Test that the block operation is correct."""
        layer = cs_layers.SmallResNetBlock(filters=4)
        input_shape = (1, 4, 4, 4)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, input_shape)

    def test_activation_effect(self):
        """Test that the activation parameter has the correct effect."""
        layer = cs_layers.SmallResNetBlock(filters=4, activation='relu')
        input_shape = (1, 4, 4, 4)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertAllGreaterEqual(output, 0.0)

    def test_batch_normalization_effect(self):
        """
        Test that batch normalization has the correct effect.
        The output the the *residual* path should be normalized.
        """
        layer = cs_layers.SmallResNetBlock(filters=4, activation=None)
        input_shape = (10000, 4, 4, 4)
        input_data = tf.random.uniform(input_shape)
        output = layer(input_data, training=True)
        res_out = output - input_data
        means = tf.reduce_mean(res_out, axis=(0,1,2))
        stds = tf.math.reduce_std(res_out, axis=(0,1,2))
        self.assertAllClose(means, tf.zeros_like(means), atol=3e-1)
        self.assertAllClose(stds, tf.ones_like(stds), atol=3e-1)

    def test_downsample_effect(self):
        """Test that the downsample parameter has the correct effect."""
        layer = cs_layers.SmallResNetBlock(filters=4, downsample=True)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 2, 2, 4))

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.SmallResNetBlock(filters=4)
        input_shape = (1, 4, 4, 4)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

# ------------------------------------------------------------------------------
        
class TestSmallDeResNetBlock(tf.test.TestCase):
            
    def setUp(self):
        pass

    def test_base_block(self):
        """Test that the block operation is correct."""
        layer = cs_layers.SmallDeResNetBlock(filters=4)
        input_shape = (1, 4, 4, 4)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, input_shape)

    def test_activation_effect(self):
        """Test that the activation parameter has the correct effect."""
        layer = cs_layers.SmallDeResNetBlock(filters=4, activation='relu')
        input_shape = (1, 4, 4, 4)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertAllGreaterEqual(output, 0.0)

    def test_batch_normalization_effect(self):
        """
        Test that batch normalization has the correct effect.
        The output the the *residual* path should be normalized.
        """
        layer = cs_layers.SmallDeResNetBlock(filters=4, activation=None)
        input_shape = (10000, 4, 4, 4)
        input_data = tf.random.uniform(input_shape)
        output = layer(input_data, training=True)
        res_out = output - input_data
        means = tf.reduce_mean(res_out, axis=(0,1,2))
        stds = tf.math.reduce_std(res_out, axis=(0,1,2))
        self.assertAllClose(means, tf.zeros_like(means), atol=3e-1)
        self.assertAllClose(stds, tf.ones_like(stds), atol=3e-1)

    def test_upsample_effect(self):
        """Test that the upsample parameter has the correct effect."""
        layer = cs_layers.SmallDeResNetBlock(filters=8, upsample=True)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 8, 8, 4))

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.SmallDeResNetBlock(filters=4)
        input_shape = (1, 4, 4, 4)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

# ------------------------------------------------------------------------------
        
class TestResNetBlock(tf.test.TestCase):
                
    def setUp(self):
        pass

    def test_base_block(self):
        """Test that the block operation is correct."""
        layer = cs_layers.ResNetBlock(filters=8)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, input_shape)

    def test_activation_effect(self):
        """Test that the activation parameter has the correct effect."""
        layer = cs_layers.ResNetBlock(filters=8, activation='relu')
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertAllGreaterEqual(output, 0.0)

    def test_batch_normalization_effect(self):
        """
        Test that batch normalization has the correct effect.
        The output the the *residual* path should be normalized.
        """
        layer = cs_layers.ResNetBlock(filters=8, activation=None)
        input_shape = (10000, 4, 4, 8)
        input_data = tf.random.uniform(input_shape)
        output = layer(input_data, training=True)
        res_out = output - input_data
        means = tf.reduce_mean(res_out, axis=(0,1,2))
        stds = tf.math.reduce_std(res_out, axis=(0,1,2))
        self.assertAllClose(means, tf.zeros_like(means), atol=3e-1)
        self.assertAllClose(stds, tf.ones_like(stds), atol=3e-1)

    def test_downsample_effect(self):
        """Test that the downsample parameter has the correct effect."""
        layer = cs_layers.ResNetBlock(filters=4, downsample=True)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 2, 2, 4))

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.ResNetBlock(filters=8)
        input_shape = (1, 4, 4, 8)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

# ------------------------------------------------------------------------------
        
class TestDeResNetBlock(tf.test.TestCase):
                        
    def setUp(self):
        pass

    def test_base_block(self):
        """Test that the block operation is correct."""
        layer = cs_layers.DeResNetBlock(filters=4)
        input_shape = (1, 4, 4, 4)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, input_shape)

    def test_activation_effect(self):
        """Test that the activation parameter has the correct effect."""
        layer = cs_layers.DeResNetBlock(filters=4, activation='relu')
        input_shape = (1, 4, 4, 4)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertAllGreaterEqual(output, 0.0)

    def test_batch_normalization_effect(self):
        """
        Test that batch normalization has the correct effect.
        The output the the *residual* path should be normalized.
        """
        layer = cs_layers.DeResNetBlock(filters=4, activation=None)
        input_shape = (10000, 4, 4, 4)
        input_data = tf.random.uniform(input_shape)
        output = layer(input_data, training=True)
        res_out = output - input_data
        means = tf.reduce_mean(res_out, axis=(0,1,2))
        stds = tf.math.reduce_std(res_out, axis=(0,1,2))
        self.assertAllClose(means, tf.zeros_like(means), atol=3e-1)
        self.assertAllClose(stds, tf.ones_like(stds), atol=3e-1)

    def test_upsample_effect(self):
        """Test that the upsample parameter has the correct effect."""
        layer = cs_layers.DeResNetBlock(filters=8, upsample=True)
        input_shape = (1, 4, 4, 8)
        input_data = tf.random.normal(input_shape)
        output = layer(input_data)
        self.assertEqual(output.shape, (input_shape[0], 8, 8, 4))

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        layer = cs_layers.DeResNetBlock(filters=4)
        input_shape = (1, 4, 4, 4)
        output_shape = layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, input_shape)

# ------------------------------------------------------------------------------
        
class TestReparameterizationLayer(tf.test.TestCase):

    def setUp(self):
        self.mean_logstds = tf.random.uniform((1,10))
        self.input_data = tf.repeat(self.mean_logstds, 10000, axis=0)
        self.layer = cs_layers.ReparameterizationLayer(5)

    def test_outputs_correct(self):
        """Test that the outputs are correct."""
        logstds, means, output = self.layer(self.input_data)
        pred_means = tf.math.reduce_mean(output, axis=0)
        pred_stds = tf.math.reduce_std(output, axis=0)
        self.assertAllEqual(logstds, self.input_data[:,5:])
        self.assertAllEqual(means, self.input_data[:,:5])
        self.assertAllClose(pred_means, self.mean_logstds[0,:5], atol=1e-1)
        self.assertAllClose(pred_stds, 
                            tf.exp(self.mean_logstds[0,5:]), atol=1e-1)

    def test_output_shape(self):
        """Test that the output shape is correct."""
        input_data = tf.random.normal((7,10))
        logstds, means, output = self.layer(input_data)
        self.assertEqual(output.shape, (7,5))
        self.assertEqual(logstds.shape, (7,5))
        self.assertEqual(means.shape, (7,5))

    def test_compute_output_shape(self):
        """Test that the computed output shape is correct."""
        input_shape = (7,10)
        output_shape = self.layer.compute_output_shape(input_shape)
        self.assertEqual(output_shape, (7,5))

# ==============================================================================
# ==============================================================================
# ============================================================================== 
        
def main():
    test_cases = (TestAWGNLayer,
                  TestEuclideanDistanceLayer,
                  TestGaussianSimilarityLayer,
                  TestSoftGaussSimPredictionLayer,
                  TestConvolutionBlock,
                  TestDeconvolutionBlock,
                  TestHeightWidthSliceLayer,
                  TestSmallResNetBlock,
                  TestSmallDeResNetBlock,
                  TestResNetBlock,
                  TestDeResNetBlock,
                  TestReparameterizationLayer)

    def load_tests(loader, tests):
        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = loader.loadTestsFromTestCase(test_class)
            filtered_tests = [t for t in tests if not \
                               t.id().endswith('.test_session')]
            suite.addTests(filtered_tests)
        return suite
    
    runner = unittest.TextTestRunner(verbosity=1)
    loader = unittest.TestLoader()
    suite = load_tests(loader, None)
    runner.run(suite)

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ------------------------------------------------------------------------------