import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
import modules.relevance_propagation as relprop
from modules.feature_extraction.mlp_feature_extractor import MlpFeatureExtractor
import scipy.special

logger = logging.getLogger("elm")


class ElmFeatureExtractor(MlpFeatureExtractor):

    def __init__(self, samples, cluster_indices, n_nodes, n_splits=10, n_iterations=10, scaling=True,
                 filter_by_distance_cutoff=False, contact_cutoff=0.5,
                 activation=relprop.relu,
                 alpha=1, name="ELM"):
        MlpFeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                     scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff,
                                     contact_cutoff=contact_cutoff,
                                     activation=activation,
                                     name="ELM")
        logger.debug("Initializing ELM with the following parameters: \
                      n_splits %s, n_iterations %s, scaling %s, filter_by_distance_cutoff %s, contact_cutoff %s, \
                      n_nodes %s, activation function %s, alpha %s", \
                      n_splits, n_iterations, scaling, filter_by_distance_cutoff, contact_cutoff, \
                      n_nodes, activation, alpha)
        self.n_nodes = n_nodes
        self.alpha = alpha

    def train(self, train_set, train_labels):
        logger.debug("Training ELM with %s samples and %s features ...", train_set.shape[0], train_set.shape[1])
        elm = SingleLayerELMClassifier(n_nodes=self.n_nodes,
                                       activation_func=self.activation,
                                       alpha=self.alpha)

        elm.fit(train_set, train_labels)
        return elm


class SingleLayerELMClassifier(object):
    def __init__(self, n_nodes, activation_func=relprop.relu, alpha=1):
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.coefs_ = None
        self.intercepts_ = None
        self.alpha = alpha # regularization constant

    def fit(self, x, t):
        (N, n) = x.shape
        W1 = self._random_matrix(n)
        b1 = self._random_matrix(1)
        H = self._g_ELM(np.matmul(x, W1) + b1)
        W2 = np.matmul(self._pseudo_inverse(H), t)
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, np.zeros((1, t.shape[1]))]

    def _random_matrix(self, x):
        return np.random.normal(0, 0.25, (x, self.n_nodes))

    def _g_ELM(self, x):
        if self.activation_func == relprop.relu: # good if you use regularization
            Z = x > 0
            return x * Z
        elif self.activation_func == relprop.logistic_sigmoid:
            return scipy.special.expit(x)
        else:
            raise Exception("Currently supported activation functions are only relu and logistic")

    def _pseudo_inverse(self, x):
    # see eq 3.17 in bishop
        try:
            inner = np.matmul(x.T, x)
            if self.alpha is not None:
                tikonov_matrix = np.eye(x.shape[1]) * self.alpha
                inner += np.matmul(tikonov_matrix.T, tikonov_matrix)
            inv = np.linalg.inv(inner)
            return np.matmul(inv, x.T)
        except np.linalg.linalg.LinAlgError as ex:
            logger.debug("inner is a singular matrix")
            # Moore Penrose inverse rule, see paper on ELM
            inner = np.matmul(x, x.T)
            if self.alpha is not None:
                tikonov_matrix = np.eye(x.shape[0]) * self.alpha
                inner += np.matmul(tikonov_matrix, tikonov_matrix.T)
            inv = np.linalg.inv(inner)
            return np.matmul(x.T, inv)

    def predict(self, x):
        H = self._g_ELM(np.matmul(x, self.coefs_[0]) + self.intercepts_[0])
        t = np.matmul(H, self.coefs_[1])
        for row_idx, row in enumerate(t):
            c_idx = row.argmax()
            t[row_idx, :] = 0
            t[row_idx, c_idx] = 1

        return t
