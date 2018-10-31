from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt

logger = logging.getLogger("postprocessing")

def vis_feature_importance(x_val, y_val, std_val, ax, extractor_name, color):
    ax.plot(x_val, y_val, color=color, label=extractor_name)
    ax.plot(x_val, y_val + std_val, color=color, markersize=1)
    ax.plot(x_val, y_val - std_val, color=color, markersize=1)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Importance")

def vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel,extractor_name, color):
    ax.bar(x_val, y_val, label=extractor_name, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def visualize(postprocessors):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    n_feature_extractors = len(postprocessors)
    cols = np.asarray([[0,0,0],[0.5,0,0],[0,0.5,0],[0,0,0.5], [0,0.5,0.5]]) #TODO: pick better colors

    fig1, axes1= plt.subplots(1,n_feature_extractors, figsize=(16,8))
    fig2, axes2 = plt.subplots(1,3, figsize=(16,8))

    counter = 0
    for pp, ax in zip(postprocessors, fig1.axes):
        #plt.plot(pp.index_to_resid, pp.importance_per_residue, label=pp.extractor.name)

        vis_feature_importance(pp.index_to_resid, pp.importance_per_residue, pp.std_importance_per_residue,
                               ax, pp.extractor.name, cols[counter, :])


        vis_performance_metrics(counter, pp.entropy, fig2.axes[0], 'Estimator', 'Relevance entropy',
                                pp.extractor.name, cols[counter, :])

        vis_performance_metrics(counter, pp.average_std, fig2.axes[1], 'Estimator', 'Average standard deviation',
                                pp.extractor.name, cols[counter, :])

        vis_performance_metrics(counter, pp.test_set_errors, fig2.axes[2], 'Estimator', 'Test set error',
                                pp.extractor.name,cols[counter, :])

        counter+=1
    plt.legend()
    plt.show()
