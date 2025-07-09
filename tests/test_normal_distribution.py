import unittest
import numpy as np
from probpipe.distributions import NormalDistribution

class TestNormalDistribution(unittest.TestCase):
    def setUp(self):
        self.mean = 0.0
        self.std_dev = 1.0
        self.dist = NormalDistribution(self.mean, self.std_dev)

    def test_sample_shape_and_type(self):
        n_samples = 10
        samples = self.dist.sample(n_samples)
        self.assertIsInstance(samples, np.ndarray)
        self.assertEqual(samples.shape, (n_samples,))

    def test_log_prob_output(self):
        data = np.array([0.0, 1.0, -1.0])
        log_probs = self.dist.log_prob(data)
        self.assertIsInstance(log_probs, np.ndarray)
        self.assertEqual(log_probs.shape, data.shape)

    def test_std_property(self):
        self.assertEqual(self.dist.std_dev, self.std_dev)

    #def test_repr_contains_mean_std(self):
    #    s = repr(self.dist)
    #    self.assertIn(str(self.mean), s)
    #    self.assertIn(str(self.std_dev), s)

    # def test_expectation_returns_normal_distribution(self):
    #     # Define a simple function g(x) = x^2
    #     def g(x: np.ndarray) -> np.ndarray:
    #         return x ** 2

    #     result_dist = self.dist.expectation(g, n_samples=5000, n_iterations=100)

    #     # Check that result is a NormalDistribution instance
    #     self.assertIsInstance(result_dist, NormalDistribution)

    #     # Check mean and std_dev are floats
    #     self.assertIsInstance(result_dist.mean, float)
    #     self.assertIsInstance(result_dist.std_dev, float)

    #     # The mean should be close to the theoretical expectation of x^2 for N(0,1), which is 1
    #     self.assertAlmostEqual(result_dist.mean, 1.0, places=2)  

    #     # The std_dev should be non-negative
    #     self.assertGreaterEqual(result_dist.std_dev, 0.0)

if __name__ == "__main__":
    unittest.main()