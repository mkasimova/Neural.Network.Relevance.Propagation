import numpy as np

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
from modules import heatmapping_modules as hm_modules

relu = "ReLu"
logistic_sigmoid = "logistic-sigmoid"
activation_functions = [relu, logistic_sigmoid]
logger = logging.getLogger("mlp")

class RelevancePropagator(object):

    def __init__(self, activation_function=relu):
        self.activation_function = activation_function


    def propagate(self, weights, biases, X, T):
        # Reinstantiating the neural network
        layers = create_layers(weights, biases, self.activation_function)
        network = Network(layers)
        Y = network.forward(X)
        # Performing relevance propagation
        D = network.relprop(Y * T)
        return D


class Network(hm_modules.Network):
    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R


class ReLU(hm_modules.ReLU):
    def relprop(self, R):
        return R

class Logistic(hm_modules.Logistic):
    def relprop(self, R):
        #TODO
        return R

class Linear(hm_modules.Linear):
    def __init__(self, weight, bias):
        self.W = weight
        self.B = bias


def create_layers(weights, biases, activation_function):
    layers = []
    for idx, weight in enumerate(weights):
        if idx == 0:
            l = FirstLinear(weight, biases[idx])
        elif activation_function == logistic_sigmoid:
            l = LogisticSigmoidLinear(weight, biases[idx])
        elif activation_function == relu:
            l = NextLinear(weight, biases[idx])
        else:
            raise Exception("Unsupported activation function {}. Supported values are {}".format(activation_function, activation_functions))
        layers.append(l)
        # ADD RELU TO THE LAST LAYER IF NEEDED
        if idx < len(weights) - 1:
            if activation_function == logistic_sigmoid:
                layers.append(Logistic())
            elif activation_function == relu:
                layers.append(ReLU())
    return layers


class FirstLinear(Linear):
    """For z-beta rule"""

    def relprop(self, R):
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + np.min(self.X, axis=0), self.X * 0 + np.max(self.X, axis=0)
        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        Wt = W if isinstance(W, int) or isinstance(W, float) else W.T #a constant just corresponds to a diagonal matrix with that constant along the diagonal
        R = X * np.dot(S, Wt) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
        return R


class LogisticSigmoidLinear(Linear):
    """For z-beta rule"""

    def relprop(self, R):
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + 1./(1+np.exp(-1)), self.X * 0 + 1. #np.max(self.X, axis=0)
        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        Wt = W if isinstance(W, int) or isinstance(W, float) else W.T #a constant just corresponds to a diagonal matrix with that constant along the diagonal
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
