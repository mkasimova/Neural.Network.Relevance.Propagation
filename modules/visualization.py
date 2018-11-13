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
    ax.plot(x_val, y_val, color=color, label=extractor_name, linewidth=2)
    ax.fill_between(x_val, y_val - std_val, y_val + std_val, color=color, alpha=0.2)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Importance")
    ax.legend()


def vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel, extractor_name, color, show_legends=False):
    ax.bar(x_val, y_val, label=extractor_name, color=color, edgecolor=color, alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legends:
        ax.legend()


def visualize(postprocessors,
              show_importance=True,
              show_performance=True):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    n_feature_extractors = len(postprocessors)
    cols = np.asarray([[0.8, 0, 0.2], [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.8,0.4,0.2],[0, 0, 0]])  # TODO: pick better colors

    if show_importance:
        fig1, axes1 = plt.subplots(1, n_feature_extractors, figsize=(35, 5))
    if show_performance:
        if postprocessors[0].predefined_relevant_residues is None:
            fig2, axes2 = plt.subplots(1, 3, figsize=(28, 5))
        else:
            fig2, axes2 = plt.subplots(1, 5, figsize=(35, 5))

    counter = 0
    for pp, ax in zip(postprocessors, fig1.axes):
        # plt.plot(pp.index_to_resid, pp.importance_per_residue, label=pp.extractor.name)
        if show_importance:
            vis_feature_importance(pp.index_to_resid, pp.importance_per_residue, pp.std_importance_per_residue,
                                   ax, pp.extractor.name, cols[counter, :])

        if show_performance:
            vis_performance_metrics(counter, pp.entropy, fig2.axes[0], 'Estimator', 'Relevance entropy',
                                    pp.extractor.name, cols[counter, :])

            vis_performance_metrics(counter, pp.average_std, fig2.axes[1], 'Estimator', 'Average standard deviation',
                                    pp.extractor.name, cols[counter, :])

            vis_performance_metrics(counter, pp.test_set_errors, fig2.axes[2], 'Estimator', 'Test set error',
                                    pp.extractor.name, cols[counter, :])

            if pp.predefined_relevant_residues is not None:
                vis_performance_metrics(counter, pp.correct_relevance_peaks, fig2.axes[3], 'Estimator',
                                        'Number of correctly predicted relevances',
                                        pp.extractor.name, cols[counter, :])

                vis_performance_metrics(counter, pp.false_positives, fig2.axes[4], 'Estimator',
                                        'Number of false positives',
                                        pp.extractor.name, cols[counter, :], show_legends=True)
        counter += 1
    # plt.legend()
    plt.show()
