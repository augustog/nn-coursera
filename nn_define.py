__author__ = 'augusto'

import numpy as np


class NN(object):
    """
    Neural network class
    __init__ (or make) create the network
    fit
    """
    #TODO: Implement sparse matrix usage and flooring of weights bellow some threshold for sparsely connected NNs
    _repr_deep = False


    def __init__(self):
        super(NN, self).__init__()

    def make(self, units, use_sparse_matrix=False):
        """
        Takes as sole argument a list, tuple, np.array or any acceptable argument for np.array()
        which contains the number of units in each layer. E.G.: [3,3,4,1] would make a NN with
        3 input layers, 3 hidden in first layer, 4 hidden in second layer and 1 output layer.

        Use sparse matrix: Choose how to internally represent interconnection matrices, np.array (cost effective
        for dense matrices, where every unit is highly connected to most units like in complex regressions) or scipy
        sparse matrices (cost effective for nns that enforce local connectivity as in image recognition NNs).
        """

        units = np.array(units)     #Convert to np.array if not already
        self.units = units

        #Next, define internal data structures
        #First, the vectors that will hold the value for each neuron
        #Second, the matrices that connect each layer

        mat_dims = np.concatenate((
                                      units[1:, np.newaxis],
                                      units[:-1, np.newaxis],
                                  ),
                                  axis=1
        )

        # Get a matrix holding shapes of all matrices that will compose the NN
        # For the example given above, a NN with [3,3,4,1] units will have matrices of shapes
        #       (3,3)
        #       (3,4)
        #       (4,1)
        #

        layers = []
        for i in units:
            layers.append(np.zeros(i, dtype=np.float64))

        mats = []
        for shape in map(tuple, mat_dims):
            mats.append(np.ones(shape, dtype=np.float64))

        self._layer_values_vectors = layers
        self._layer_interconnection_matrices = mats

        self.input_shape = (units[0].astype(np.int),)

    #Interface for compatibility with scikit-learn

    def predict(self, x):
        """
        Predict y from input x
        """
        layers = len(self._layer_values_vectors)
        try:
            if x.shape != self.input_shape:
                raise ValueError("incompatible dimensions")
        except AttributeError:
            raise ValueError("x must be np.array")

        self._layer_values_vectors[0] = x
        for idx in xrange(layers - 1):
            self._layer_values_vectors[idx + 1] = (
                np.dot(
                    self._layer_interconnection_matrices[idx],
                    self._layer_values_vectors[idx]
                )
            )
        return self._layer_values_vectors[-1]

    def fit(self, X, y):
        """
        Fit the model to the training data.
        X is training input
        y is desired output
        """
        try:
            x_shape = X.shape
            y_shape = y.shape
        except AttributeError:
            raise ValueError("x must be np.array")
        desired_X_shape = tuple([y_shape, self.input_shape])

        if x_shape == desired_X_shape:
            sample_axis = 0
        elif x_shape[::-1] == desired_X_shape:
            sample_axis = 1
        else:
            raise ValueError("incompatible dimensions")

        #TODO: Check if this can be avoided, big datasets will appreciate
        if sample_axis == 1:
            X = np.transpose(X)
            y = np.transpose(y)
        self._fit(X, y)
        return self

    #Representation control

    def __repr__(self):
        if self._repr_deep:
            representation = ""
            n = 0
            for mat in self._layer_interconnection_matrices:
                representation += """
                Layer {} to {}
                ===============
                """.format(n, n + 1) + repr(mat) + "\n\n"
                n += 1
        else:
            representation = "<NeuralNetwork NN instance of shape {}, >".format(str(self.units))
        return representation

    @classmethod
    def repr_deep(cls, represent_deep=True):
        cls._repr_deep = represent_deep

    #Internals
    def _fit(self, X, y):
        pass



