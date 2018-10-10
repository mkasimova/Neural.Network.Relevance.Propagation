import numpy as np
import Modules


def relevance_propagation(weights, biases, X, T):
    # Reinstantiating the neural network
    network = Network(create_layers(weights, biases))
    Y = network.forward(X)
    # Performing relevance propagation
    D = network.relprop(Y * T)
    return D


class Network(Modules.Network):
    def relprop(self, R):
        for l in self.layers[::-1]:
            R = l.relprop(R)
        return R


class ReLU(Modules.ReLU):
    def relprop(self, R):
        return R


class Linear(Modules.Linear):
    def __init__(self, weight, bias):
        self.W = weight
        self.B = bias


def create_layers(weights, biases, use_first_linear=True):
    layers = []
    for idx, weight in enumerate(weights):
        if idx == 0 and use_first_linear:
            l = FirstLinear(weight, biases[idx])
        else:
            l = NextLinear(weight, biases[idx])
        layers.append(l)
        # ADD RELU TO THE LAST LAYER IF NEEDED
        if idx < len(weights) - 1:
            layers.append(ReLU())
    return layers


class FirstLinear(Linear):
    """For z-beta rule"""
    def relprop(self, R):
        W, V, U = self.W, np.maximum(0, self.W), np.minimum(0, self.W)
        X, L, H = self.X, self.X * 0 + np.min(self.X,axis=0), self.X * 0 + np.max(self.X,axis=0)
        Z = np.dot(X, W) - np.dot(L, V) - np.dot(H, U) + 1e-9
        S = R / Z
        R = X * np.dot(S, W.T) - L * np.dot(S, V.T) - H * np.dot(S, U.T)
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
