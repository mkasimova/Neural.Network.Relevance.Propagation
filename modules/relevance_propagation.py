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
activation_functions = [relu, logistic_sigmoid]
logger = logging.getLogger("mlp")


class RelevancePropagator(object):

    def __init__(self, layers, activation_function=relu):
        self.activation_function = activation_function
        self.layers = layers

    def propagate(self, X, T):
        # Reinstantiating the neural network
        network = Network(self.layers)
        Y = network.forward(X)
        # Performing relevance propagation
        D = network.relprop(Y * T)
        return D


class Network(hm_modules.Network): #TODO move forward from heatmapping_modules.py? remove heatmapping.py?
    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R


class ReLU(hm_modules.ReLU): #TODO move forward from heatmapping.py?
    def relprop(self, R):
        return R


class LogisticSigmoid:
    """Used for RBM"""

    def __init__(self): #TODO remove?
        pass

    @staticmethod
    def logistic(X):
        # return 1.0 / (1.0 + np.exp(-X))
        return scipy.special.expit(X)

    def forward(self, X):
        # self.X = X
        return self.logistic(X)

    def gradprop(self, DY): #TODO remove?
        # self.DY = DY
        # TODO double check implementation
        l = self.logistic(DY)
        return l * (1 - l)

    def relprop(self, R):
        return R


class Linear(hm_modules.Linear): #TODO remove?
    def __init__(self, weight, bias):
        self.W = weight
        self.B = bias


class FirstLinear(Linear): #TODO remove Linear? move forward from heatmapping.py?
    """For z-beta rule"""

    def relprop(self, R):
        min_val, max_val = np.min(self.X, axis=0), np.max(self.X, axis=0)
        if min_val.min() < 0: #TODO do we need this condition (and the following)?
            logger.warn("Expected input to be scaled between 0 and 1. Minimum value was %s", min_val)
        else:
            min_val = 0
        if max_val.max() > 1 + 1e-3: #TODO remove 1e-3?
            logger.warn("Expected input to be scaled between 0 and 1. Max value was %s", max_val)
        else:
            max_val = 1
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + min_val, self.X * 0 + max_val
        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        Wt = W if isinstance(W, int) or isinstance(W, float) else W.T # a constant just corresponds to a diagonal matrix with that constant along the diagonal #TODO why do we need this condition?
        R = X * np.dot(S, Wt) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
        return R


class NextLinear(Linear): #TODO remove Linear? move forward from heatmapping.py?
    """For z+ rule"""

    def relprop(self, R):
        V = np.maximum(0, self.W)
        Z = np.dot(self.X, V) + 1e-9
        S = R / Z
        C = np.dot(S, V.T)
        R = self.X * C
        return R
