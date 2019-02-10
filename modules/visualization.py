from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np
from modules import postprocessing as pop, utils

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt

logger = logging.getLogger("visualization")


def _vis_feature_importance(x_val, y_val, std_val, ax, extractor_name, color, average=None):
    y_val, std_val = y_val.squeeze(), std_val.squeeze() #Remove unnecessary unit dimensions for visualization
    ax.plot(x_val, y_val, color=color, label=extractor_name, linewidth=2)
    ax.fill_between(x_val, y_val - std_val, y_val + std_val, color=color, alpha=0.2)
    if average is not None:
        ax.plot(x_val,average,color='k',linestyle='--',label="Feature extractor average")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Importance")
    ax.legend()


def _vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel, extractor_name, color, show_legends=False,std_val=None):
    ax.plot(x_val, y_val, label=extractor_name, color=color, linewidth=2, marker='o',markersize=5)
    if std_val is not None:
        ax.plot([x_val,x_val], [y_val-std_val,y_val+std_val], color='k', linewidth=1, linestyle='-',marker='s',markersize=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legends:
        ax.legend()

def _vis_per_cluster_projection_entropy(x_val, y_val, width, ax, col, extractor_name, std_val=None, xlabel='',ylabel='',ylim=None):
    ax.bar(x_val,y_val,width, color=col,edgecolor='',label=extractor_name)
    if std_val is not None:
        for i in range(x_val.shape[0]):
            ax.plot([x_val[i],x_val[i]],[y_val[i] - std_val[i], y_val[i] + std_val[i]], color='k', linewidth=1, linestyle='-',marker='s',markersize=1)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return

def _vis_multiple_run_performance_metrics(x_vals, metrics, metric_labels, per_cluster_projection_entropies,
                                         extractor_names, colors):
    """
    Visualize (average + stddev) performance metrics of multiple runs.
    :param x_vals:
    :param metrics:
    :param metric_labels:
    :param per_cluster_projection_entropies:
    :param extractor_names:
    :param colors:
    :return:
    """
    n_estimators = metrics[0].shape[0]
    n_metrics = len(metrics)

    width = 1.0 / n_estimators - 0.05

    ave_metrics = []
    std_metrics = []

    for i in range(n_metrics):
        ave_metrics.append(metrics[i].mean(axis=1))
        std_metrics.append(metrics[i].std(axis=1))

    ave_per_cluster_projection_entropies = per_cluster_projection_entropies.mean(axis=1)
    std_per_cluster_projection_entropies = per_cluster_projection_entropies.std(axis=1)

    cluster_proj_entroy_ylim = [0,np.max(ave_per_cluster_projection_entropies+std_per_cluster_projection_entropies+0.1)]

    x_val_clusters = np.arange(ave_per_cluster_projection_entropies.shape[1])-width*n_estimators/2.0

    fig1, _ = plt.subplots(1, n_metrics, figsize=(38, 5))
    fig2, _ = plt.subplots(1, 1, figsize=(20, 5))

    for i_metric in range(n_metrics):
        fig1.axes[i_metric].plot(x_vals, ave_metrics[i_metric],color=[0.4,0.4,0.45],linewidth=1)

    for i_estimator in range(n_estimators):
        # Visualize each performance metric for current estimator with average+-std, in each axis
        show_legends = False
        for i_metric in range(n_metrics):
            if i_metric == n_metrics -1:
                show_legends = True
            _vis_performance_metrics(x_vals[i_estimator], ave_metrics[i_metric][i_estimator], fig1.axes[i_metric], 'Estimator',
                                    metric_labels[i_metric], extractor_names[i_estimator],
                                    colors[i_estimator], std_val=std_metrics[i_metric][i_estimator],show_legends=show_legends)

        _vis_per_cluster_projection_entropy(x_val_clusters+width*i_estimator, ave_per_cluster_projection_entropies[i_estimator,:], width, fig2.axes[0], colors[i_estimator],
                                           extractor_names[i_estimator], std_val=std_per_cluster_projection_entropies[i_estimator,:],
                                           xlabel='Cluster',ylabel='Projection entropy',ylim=cluster_proj_entroy_ylim)
    return

def _vis_projected_data(proj_data, cluster_indices, fig, title):
    """
    Scatter plot of projected data and cluster labels.
    :param proj_data:
    :param cluster_indices:
    :param fig:
    :param title:
    :return:
    """
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
                axes[counter-1].scatter(proj_data[:,i],proj_data[:,j],s=15,c=cluster_indices,edgecolor='', alpha=0.3)
                counter += 1
    return

def get_average_feature_importance(postprocessors,i_run):
    importances = []
    std_importances = []
    for pp in postprocessors:
        importances.append(pp[i_run].importance_per_residue)
        std_importances.append(pp[i_run].std_importance_per_residue)
    importances = np.asarray(importances).mean(axis=0)
    std_importances = np.asarray(std_importances).mean(axis=0)
    importances, std_importances = utils.rescale_feature_importance(importances, std_importances)
    return importances, std_importances

def extract_metrics(postprocessors):
    """
    Extract performance metrics from multiple runs.
    :param postprocessors:
    :param data_projection:
    :return:
    """
    n_runs = len(postprocessors[0])
    n_estimators = len(postprocessors)
    n_clusters = postprocessors[0][0].data_projector.n_clusters

    x_vals = np.arange(n_estimators)
    standard_devs = np.zeros((n_estimators,n_runs))
    test_set_errors = np.zeros((n_estimators,n_runs))
    separation_scores = np.zeros((n_estimators,n_runs))
    projection_entropies = np.zeros((n_estimators,n_runs))
    per_cluster_projection_entropies = np.zeros((n_estimators, n_runs, n_clusters))
    extractor_names = []

    for i_run in range(n_runs):
        for i_estimator in range(n_estimators):
            pp = postprocessors[i_estimator][i_run]
            standard_devs[i_estimator,i_run] = pp.average_std
            test_set_errors[i_estimator,i_run] = pp.test_set_errors
            separation_scores[i_estimator,i_run] = pp.data_projector.separation_score
            projection_entropies[i_estimator,i_run] = pp.data_projector.projection_class_entropy
            per_cluster_projection_entropies[i_estimator,i_run,:] = pp.data_projector.cluster_projection_class_entropy
            if i_run == 0:
                extractor_names.append(pp.extractor.name)

    metric_labels = ['Average standard deviation','Average test set error',
                     'Separation score','Projection entropy']

    metrics = [standard_devs, test_set_errors, separation_scores, projection_entropies]

    return x_vals, metrics, metric_labels, per_cluster_projection_entropies, extractor_names

def visualize(postprocessors, show_importance=True, show_performance=True, show_projected_data=False):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    n_feature_extractors = len(postprocessors)
    cols = np.asarray([[0.7, 0.0, 0.08], [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.8, 0.6, 0.2], [1, 0.8, 0]])

    if show_performance:
        x_vals, metrics, metric_labels, per_cluster_projection_entropies, extractor_names = extract_metrics(postprocessors)

        _vis_multiple_run_performance_metrics(x_vals, metrics, metric_labels, per_cluster_projection_entropies,
                                             extractor_names, cols)

    # Visualize the first run
    i_run = 0
    if show_importance:
        ave_feats, std_feats = get_average_feature_importance(postprocessors, i_run)
        fig1, axes1 = plt.subplots(1, n_feature_extractors, figsize=(35, 3))
        counter = 0
        for pp, ax in zip(postprocessors, fig1.axes):
            _vis_feature_importance(pp[i_run].index_to_resid, pp[i_run].importance_per_residue, pp[i_run].std_importance_per_residue,
                                   ax, pp[i_run].extractor.name, cols[counter, :], average=ave_feats)
            counter+=1


    if show_projected_data:
        fig_counter = 3
        for pp in postprocessors:
            dp = pp[i_run].data_projector
            if dp.raw_projection is not None:
                _vis_projected_data(dp.raw_projection, dp.labels, plt.figure(fig_counter), "Raw projection "+pp[i_run].extractor.name)
                fig_counter += 1

    plt.show()
