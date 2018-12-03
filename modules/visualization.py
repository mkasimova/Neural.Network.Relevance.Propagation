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

logger = logging.getLogger("visualization")


def vis_feature_importance(x_val, y_val, std_val, ax, extractor_name, color, average=None):
    ax.plot(x_val, y_val, color=color, label=extractor_name, linewidth=2)
    ax.fill_between(x_val, y_val - std_val, y_val + std_val, color=color, alpha=0.2)
    if average is not None:
        ax.plot(x_val,average,color='k',linestyle='--',label="Feature extractor average")
    ax.set_xlabel("Residue")
    ax.set_ylabel("Importance")
    ax.legend()


def vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel, extractor_name, color, show_legends=False,std_val=None):
    ax.plot(x_val, y_val, label=extractor_name, color=color, linewidth=2, marker='o',markersize=5)
    if std_val is not None:
        ax.plot([x_val,x_val], [y_val-std_val,y_val+std_val], color='k', linewidth=1, linestyle='-',marker='s',markersize=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legends:
        ax.legend()

def vis_per_cluster_projection_entropy(x_val, y_val, width, ax, col, extractor_name, std_val=None, xlabel='',ylabel='',ylim=None):
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

def vis_multiple_run_performance_metrics(x_vals, metrics, metric_labels, per_cluster_projection_entropies,
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
            vis_performance_metrics(x_vals[i_estimator], ave_metrics[i_metric][i_estimator], fig1.axes[i_metric], 'Estimator',
                                    metric_labels[i_metric], extractor_names[i_estimator],
                                    colors[i_estimator], std_val=std_metrics[i_metric][i_estimator],show_legends=show_legends)

        vis_per_cluster_projection_entropy(x_val_clusters+width*i_estimator, ave_per_cluster_projection_entropies[i_estimator,:], width, fig2.axes[0], colors[i_estimator],
                                           extractor_names[i_estimator], std_val=std_per_cluster_projection_entropies[i_estimator,:],
                                           xlabel='Cluster',ylabel='Projection entropy',ylim=cluster_proj_entroy_ylim)
    return

def vis_projected_data(proj_data, cluster_indices, fig, title):
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
                axes[counter-1].scatter(proj_data[:,i],proj_data[:,j],s=15,c=cluster_indices,edgecolor='')
                counter += 1
    return

def get_average_feature_importance(postprocessors,i_run):
    importances = []
    for pp in postprocessors:
        importances.append(pp[i_run].importance_per_residue)
    importances = np.asarray(importances).mean(axis=0)
    importances,_ = pop.rescale_feature_importance(importances, importances)
    return importances

def extract_metrics(postprocessors, data_projectors):
    """
    Extract performance metrics from multiple runs.
    :param postprocessors:
    :param data_projection:
    :return:
    """
    n_runs = len(postprocessors[0])
    n_estimators = len(postprocessors)
    n_clusters = data_projectors[0][0].n_clusters

    x_vals = np.arange(n_estimators)
    standard_devs = np.zeros((n_estimators,n_runs))
    entropies = np.zeros((n_estimators,n_runs))
    test_set_errors = np.zeros((n_estimators,n_runs))
    separation_scores = np.zeros((n_estimators,n_runs))
    projection_entropies = np.zeros((n_estimators,n_runs))
    per_cluster_projection_entropies = np.zeros((n_estimators, n_runs, n_clusters))
    extractor_names = []

    for i_run in range(n_runs):
        for i_estimator in range(n_estimators):
            standard_devs[i_estimator,i_run] = postprocessors[i_estimator][i_run].average_std
            entropies[i_estimator,i_run] = postprocessors[i_estimator][i_run].entropy
            test_set_errors[i_estimator,i_run] = postprocessors[i_estimator][i_run].test_set_errors
            separation_scores[i_estimator,i_run] = data_projectors[i_estimator][i_run].separation_score
            projection_entropies[i_estimator,i_run] = data_projectors[i_estimator][i_run].projection_class_entropy
            per_cluster_projection_entropies[i_estimator,i_run,:] = data_projectors[i_estimator][i_run].cluster_projection_class_entropy
            if i_run == 0:
                extractor_names.append(postprocessors[i_estimator][i_run].extractor.name)

    metric_labels = ['Average standard deviation','Average relevance entropy','Average test set error',
                     'Separation score','Projection entropy']

    metrics = [standard_devs, entropies, test_set_errors, separation_scores, projection_entropies]

    return x_vals, metrics, metric_labels, per_cluster_projection_entropies, extractor_names

def extract_test_projection_entropy(data_projectors):
    n_runs = len(data_projectors[0])
    n_estimators = len(data_projectors)
    data = np.zeros((n_estimators, n_runs, data_projectors[0][0].test_projection_class_entropy.shape[0]))
    for i_run in range(n_runs):
        for i_estimator in range(n_estimators):
            data[i_estimator,i_run,:] = data_projectors[i_estimator][i_run].test_projection_class_entropy
    return data.mean(axis=1), data.std(axis=1)

def visualize(postprocessors, data_projectors, show_importance=True, show_performance=True, show_projected_data=False):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    n_feature_extractors = len(postprocessors)
    cols = np.asarray([[0.7, 0.0, 0.08], [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.8, 0.6, 0.2], [1, 0.8, 0]])

    if show_performance:
        x_vals, metrics, metric_labels, per_cluster_projection_entropies, extractor_names = extract_metrics(postprocessors, data_projectors)

        vis_multiple_run_performance_metrics(x_vals, metrics, metric_labels, per_cluster_projection_entropies,
                                             extractor_names, cols)

    # Visualize the first run
    i_run = 0
    if show_importance:
        ave_feats = get_average_feature_importance(postprocessors, i_run)

        fig1, axes1 = plt.subplots(1, n_feature_extractors, figsize=(35, 3))
        counter = 0
        for pp, ax in zip(postprocessors, fig1.axes):
            vis_feature_importance(pp[i_run].index_to_resid, pp[i_run].importance_per_residue, pp[i_run].std_importance_per_residue,
                                   ax, pp[i_run].extractor.name, cols[counter, :], average=ave_feats)
            counter+=1


    if show_projected_data:
        fig_counter = 3
        for pp, dp in zip(postprocessors, data_projectors):
            if dp.raw_projection is not None:
                vis_projected_data(dp[i_run].raw_projection, dp[i_run].labels, plt.figure(fig_counter), "Raw projection "+dp[i_run].extractor.name)
                fig_counter += 1

    if data_projectors[0][0].test_projection_class_entropy is not None:
        plt.figure(4)
        ave_proj_entropies, std_proj_entropies = extract_test_projection_entropy(data_projectors)
        x_val = np.arange(ave_proj_entropies.shape[1])

        for i_estimator in range(ave_proj_entropies.shape[0]):
            y_val = ave_proj_entropies[i_estimator,:]
            std_val = std_proj_entropies[i_estimator,:]
            plt.plot(x_val,y_val,color=cols[i_estimator,:],linewidth=2)
            plt.fill_between(x_val, y_val - std_val, y_val + std_val, color=cols[i_estimator,:], alpha=0.2)

    plt.show()
