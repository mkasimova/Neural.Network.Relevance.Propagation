import numpy as np

from modules import heatmapping_modules as hm_modules


def relevance_propagation(weights, biases, X, T, use_sigmoid_activation=False):
    # Reinstantiating the neural network
    network = Network(create_layers(weights, biases, use_sigmoid_activation=use_sigmoid_activation))
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
        return R

class Linear(hm_modules.Linear):
    def __init__(self, weight, bias):
        self.W = weight
        self.B = bias


def create_layers(weights, biases, use_first_linear=True, use_sigmoid_activation=False):
    layers = []
    for idx, weight in enumerate(weights):
        if (idx == 0 and use_first_linear) or use_sigmoid_activation:
            l = FirstLinear(weight, biases[idx])
        else:
            l = NextLinear(weight, biases[idx])
        layers.append(l)
        # ADD RELU TO THE LAST LAYER IF NEEDED
        if use_sigmoid_activation:
            if idx < len(weights) - 1:
                layers.append(Logistic())
        else:
            if idx < len(weights) - 1:
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


class NextLinear(Linear):
    """For z+ rule"""

    def relprop(self, R):
        V = np.maximum(0, self.W)
        Z = np.dot(self.X, V) + 1e-9
        S = R / Z
        C = np.dot(S, V.T)
        R = self.X * C
        return R
