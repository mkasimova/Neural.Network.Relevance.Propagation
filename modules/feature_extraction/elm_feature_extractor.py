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

    def __init__(self, samples, cluster_indices, n_splits=10, n_iterations=10, scaling=True,
                 filter_by_distance_cutoff=False, contact_cutoff=0.5, 
                 activation=relprop.relu,
                 n_nodes=None, alpha=1):
        MlpFeatureExtractor.__init__(self, samples, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
                                     scaling=scaling, filter_by_distance_cutoff=filter_by_distance_cutoff,
                                     contact_cutoff=contact_cutoff, 
                                     activation=activation,
                                     name="ELM")
        self.n_nodes = n_nodes
        self.alpha = alpha

    def train(self, train_set, train_labels):
        elm = SingleLayerELMClassifier(n_nodes=self.n_nodes,
                                       activation_func=self.activation,
                                       alpha=self.alpha)

        elm.fit(train_set, train_labels)
        return elm


def pseudo_inverse(x, alpha=None):
    # see eq 3.17 in bishop
    try:
        inner = np.matmul(x.T, x)
        if alpha is not None:
            tikonov_matrix = np.eye(x.shape[1]) * alpha
            inner += np.matmul(tikonov_matrix.T, tikonov_matrix)
        inv = np.linalg.inv(inner)
        return np.matmul(inv, x.T)
    except np.linalg.linalg.LinAlgError as ex:
        logger.debug("Singular matrix")
        # Moore Penrose inverse rule, see paper on ELM
        if alpha is not None:
            tikonov_matrix = np.eye(x.shape[0]) * alpha
            inner += np.matmul(tikonov_matrix, tikonov_matrix.T)
        inv = np.linalg.inv(np.matmul(x, x.T))
        return np.matmul(x.T, inv)


def random_matrix(L, n):
    return np.random.normal(0, 0.25, (L, n))
    # return np.random.rand(L, n)


def g_ELM(x, func_name):
    if func_name == "soft_relu":  # good
        return np.log(1 + np.exp(x))
    elif func_name == relprop.relu:  # good if you use regularization
        Z = x > 0
        return x * Z
    elif func_name == "arctan":  # also good
        return np.arctan(x)
    elif func_name == "identity":
        return x
    elif func_name == relprop.logistic_sigmoid:
        return scipy.special.expit(x)
    else:
        raise Exception("No such activation function {}".format(func_name))


class SingleLayerELMClassifier(object):
    def __init__(self, n_nodes=None, activation_func="hard_relu", alpha=1.):
        self.n_nodes = n_nodes
        self.activation_func = activation_func
        self.coefs_ = None
        self.intercepts_ = None
        self.alpha = alpha  # regularization constant

    def fit(self, x, t):
        (N, n) = x.shape
        if self.n_nodes is None:
            self.n_nodes = min(4000, n)
        ####    logger.info("Automatically settings number of nodes in first layer to %s", self.n_nodes)
        W1 = random_matrix(n, self.n_nodes)
        b1 = random_matrix(1, self.n_nodes)
        H = g_ELM(np.matmul(x, W1) + b1, self.activation_func)
        W2 = np.matmul(pseudo_inverse(H, alpha=self.alpha), t)
        self.coefs_ = [W1, W2]
        self.intercepts_ = [b1, np.zeros((1, t.shape[1]))]

    def predict(self, x):
        H = g_ELM(np.matmul(x, self.coefs_[0]) + self.intercepts_[0], func_name=self.activation_func)
        t = np.matmul(H, self.coefs_[1])
        for row_idx, row in enumerate(t):
            c_idx = row.argmax()
            t[row_idx, :] = 0
            t[row_idx, c_idx] = 1

        return t
