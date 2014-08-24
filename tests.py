"""
Test suite for the functions in the neural network package
"""

__author__ = 'augusto'

import unittest
import nn_define as ndef
import nn_optimize as nopt

from itertools import izip
import numpy as np
from numpy.testing import assert_array_equal

class NNDefine (unittest.TestCase):
    def setUp(self):
        self.nn = ndef.NN()

    def test_make(self):
        self.nn.make([3, 2, 1, 2])
        mat_dims = [(2,3), (1,2), (2,1)]        #TODO: Verify mathematically
        self.assertEqual(len(self.nn._layer_interconnection_matrices), 3)
        for mat, dims in izip(
                self.nn._layer_interconnection_matrices,
                mat_dims):
            self.assertEqual(mat.shape, dims)


class NNPredict (unittest.TestCase):
    def setUp(self):
        self.nn = ndef.NN()
        self.nn.make([3, 3, 1])

    def test_predict(self):
        input = np.array([1, 1, 1])
        predicted_value = self.nn.predict(input)
        self.assertTupleEqual(predicted_value.shape, (1,))
        self.assertEqual(predicted_value[0], 9)


class NNOptimize (unittest.TestCase):
    def setUp(self):
        self.nn = ndef.NN()
        self.nn.make([3, 3, 1])

    def test_nonregularized_cost_function(self):
        """
        Test cost function
        J(Theta) = SquaredError
        given a NN like the one in NNPredict test suite, and
        three equal inputs (whose output is np.array([9]))
        and given the target values 8, 10 and 8,
        total squared error should be 3.
        """
        X = np.array([[1,1,1],[1,1,1],[1,1,1]])
        y = np.array([[8],[10],[8]])

        cost = self.nn._cost_not_regularized(X, y)
        self.assertIsInstance(cost, np.float64)
        self.assertEqual(cost, 3.)

    def test_optimize(self):
        pass


if __name__ == '__main__':
    unittest.main()