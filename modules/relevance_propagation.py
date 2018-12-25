import logging
import sys

import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
from modules import heatmapping_modules as hm_modules
import scipy.special

relu = "relu"
logistic_sigmoid = "logistic"
logger = logging.getLogger("relevance_propagation")


class RelevancePropagator(object):

    def __init__(self, layers):
        self.layers = layers

    def propagate(self, X, T):
        # Reinstantiating the neural network
        network = Network(self.layers)
        Y = network.forward(X)
        # Performing relevance propagation
        D = network.relprop(Y * T)
        return D


class Network:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, X):
        for l in self.layers: X = l.forward(X)
        return X

    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R


class ReLU:

    def forward(self, X):
        self.Z = X > 0
        return X * self.Z

    def relprop(self, R):
        return R


class LogisticSigmoid:
    """Used for RBM"""

    def _logistic(self, X):
        return scipy.special.expit(X)

    def forward(self, X):
        return self._logistic(X)

    def relprop(self, R):
        return R


class Linear:
    def __init__(self, weight, bias):
        self.W = weight
        self.B = bias

    def forward(self, X):
        self.X = X
        return np.dot(self.X, self.W) + self.B


class FirstLinear(Linear):
    """For z-beta rule"""

    def relprop(self, R):
        min_val, max_val = np.min(self.X, axis=0), np.max(self.X, axis=0)
        min_val = 0
        max_val = 1
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + min_val, self.X * 0 + max_val
        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        Wt = W if isinstance(W, int) or isinstance(W, float) else W.T # a constant just corresponds to a diagonal matrix with that constant along the diagonal
        R = X * np.dot(S, Wt) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
        return R


class NextLinear(Linear):
    """For z+ rule"""

    def relprop(self, R):
        V = np.maximum(0, self.W)
        Z = np.dot(self.X, V) + 1e-9
        S = R / Z
        C = np.dot(S, V.T)
        R = self.X * C
        return R
