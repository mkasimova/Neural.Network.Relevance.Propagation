from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np
from modules import postprocessing as pop

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt

logger = logging.getLogger("postprocessing")


def vis_feature_importance(x_val, y_val, std_val, ax, extractor_name, color, average=None):
    ax.plot(x_val, y_val, color=color, label=extractor_name, linewidth=2)
    ax.fill_between(x_val, y_val - std_val, y_val + std_val, color=color, alpha=0.2)
    if average is not None:
        ax.plot(x_val,average,color='k',linestyle='--',label="Feature extractor average")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Importance")
    ax.legend()


def vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel, extractor_name, color, show_legends=False,x_val_prev=None, y_val_prev=None):

    if x_val_prev is not None and y_val_prev is not None:
        ax.plot([x_val_prev,x_val], [y_val_prev,y_val], color='k', linewidth=1, linestyle='--')
    ax.plot(x_val, y_val, label=extractor_name, color=color, linewidth=2, marker='o',markersize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legends:
        ax.legend()

def vis_projected_data(proj_data, cluster_indices, fig, title):
    n_dims = proj_data.shape[1]
    n_combi = float(n_dims*(n_dims-1)/2)
    counter = 1
    plt.title(title)
    axes = []

    if n_dims == 1:
        plt.scatter(proj_data[:, 0],np.zeros(proj_data.shape[0]), s=15, c=cluster_indices, edgecolor='')
    else:
        plt.axis('off')
        for i in range(n_dims):
            for j in range(i+1,n_dims):
                axes.append(fig.add_subplot(np.ceil(n_combi/3),3,counter))
                axes[counter-1].scatter(proj_data[:,i],proj_data[:,j],s=15,c=cluster_indices,edgecolor='')
                counter += 1
    return

def get_average_feature_importance(postprocessors):
    importances = []
    for pp in postprocessors:
        importances.append(pp.importance_per_residue)
    importances = np.asarray(importances).mean(axis=0)
    importances,_ = pop.rescale_feature_importance(importances, importances)
    return importances

def visualize(postprocessors, data_projectors, show_importance=True, show_performance=True, show_projected_data=False):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    ave_feats = get_average_feature_importance(postprocessors)

    n_feature_extractors = len(postprocessors)
    cols = np.asarray([[0.7, 0.0, 0.08], [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.9,0.6,0.2],[0, 0, 0]])

    if show_importance:
        fig1, axes1 = plt.subplots(1, n_feature_extractors, figsize=(35, 5))

    if show_performance:
        if postprocessors[0].predefined_relevant_residues is None:
            fig2, axes2 = plt.subplots(1, 3, figsize=(28, 5))
        else:
            fig2, axes2 = plt.subplots(1, 5, figsize=(35, 5))

    counter = 0
    fig_counter = 3

    x_val_prev = None
    sep_score_prev = None
    std_prev = None
    entropy_prev= None

    for pp, dp, ax in zip(postprocessors, data_projectors, fig1.axes):
        # plt.plot(pp.index_to_resid, pp.importance_per_residue, label=pp.extractor.name)
        if show_importance:
            vis_feature_importance(pp.index_to_resid, pp.importance_per_residue, pp.std_importance_per_residue,
                                   ax, pp.extractor.name, cols[counter, :], average=ave_feats)

        if show_performance:
            vis_performance_metrics(counter, pp.entropy, fig2.axes[0], 'Estimator', 'Relevance entropy',
                                    pp.extractor.name, cols[counter, :], x_val_prev=x_val_prev,y_val_prev=entropy_prev)

            vis_performance_metrics(counter, pp.average_std, fig2.axes[1], 'Estimator', 'Average standard deviation',
                                    pp.extractor.name, cols[counter, :], x_val_prev=x_val_prev,y_val_prev=std_prev)


            if pp.predefined_relevant_residues is not None:
                vis_performance_metrics(counter, dp.separation_score, fig2.axes[2], 'Estimator', 'Separation score',
                                        pp.extractor.name, cols[counter, :], x_val_prev=x_val_prev,y_val_prev=sep_score_prev)

                vis_performance_metrics(counter, pp.correct_relevance_peaks, fig2.axes[3], 'Estimator',
                                        'Number of correctly predicted relevances',
                                        pp.extractor.name, cols[counter, :])

                vis_performance_metrics(counter, pp.false_positives, fig2.axes[4], 'Estimator',
                                        'Number of false positives',
                                        pp.extractor.name, cols[counter, :], show_legends=True)
            else:
                vis_performance_metrics(counter, dp.separation_score, fig2.axes[2], 'Estimator', 'Separation score',
                                        pp.extractor.name, cols[counter, :], show_legends=True, x_val_prev=x_val_prev,y_val_prev=sep_score_prev)


            x_val_prev = counter
            std_prev = pp.average_std
            entropy_prev = pp.entropy
            sep_score_prev = dp.separation_score

        if show_projected_data:
            if dp.raw_projection is not None:
                vis_projected_data(dp.raw_projection, dp.labels, plt.figure(fig_counter), "Raw projection "+dp.extractor.name)
                fig_counter += 1

        counter += 1
    plt.show()
