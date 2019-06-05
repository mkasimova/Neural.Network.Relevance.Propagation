from __future__ import absolute_import, division, print_function

import logging
import sys

import matplotlib.pyplot as plt

plt.style.use("seaborn-colorblind")
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import numpy as np
from . import utils

logger = logging.getLogger("visualization")
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.size'] = 18
_blue = [50.0 / 256.0, 117.0 / 256.0, 220.0 / 256.0]
_gray = [33.0 / 256.0, 36.0 / 256.0, 50.0 / 256.0]
_boxprops = dict(facecolor=_blue)


def _plot_highligthed_residues(highlighted_residues, ax, linestyles=['dashed', 'solid', 'dotted', 'dashdot'],
                               alpha=0.67, linewidth=1):
    if isinstance(highlighted_residues, dict):
        for idx, (label, residues) in enumerate(highlighted_residues.items()):
            for r_idx, r in enumerate(residues):
                ax.axvline(r, linestyle=linestyles[idx % len(linestyles)],
                           label=label if r_idx == 0 else None,
                           color=_blue,
                           linewidth=linewidth,
                           alpha=alpha)
        ax.legend()
    else:
        for r in highlighted_residues:
            ax.axvline(r, linestyle=linestyles[0], label=None, color=_blue, linewidth=linewidth,
                       alpha=alpha)


def _insert_gaps(x_val, y_val):
    """
    Look for gaps in x_val (assuming it is a sequence of integers) and insert np.nan in the correspoding position in y
    This should insert gaps in plots where residues are missing
    :param x_val:
    :param y_val:
    :return:
    """
    last_x = None
    new_x = []
    new_y = []
    eps = 1e-4
    for i, x in enumerate(x_val):
        if isinstance(x, str) or abs(np.rint(x) - x) > eps:
            # We're note dealing with integer data
            return x_val, y_val
        x = int(np.rint(x))
        if last_x is not None and x - last_x - 1 > 0:
            # Fill gaps
            for g in range(last_x + 1, x + 1):
                new_x.append(g)
                new_y.append(np.nan)
        new_x.append(x)
        new_y.append(y_val[i])
        last_x = x
    new_x = np.array(new_x)
    new_y = np.array(new_y)
    return new_x, new_y


def _vis_feature_importance(x_val, y_val, std_val, ax, extractor_name, color, average=None, highlighted_residues=None,
                            show_title=True, set_ylim=True):
    x_val, y_val = _insert_gaps(x_val, y_val)
    y_val, std_val = y_val.squeeze(), std_val.squeeze()  # Remove unnecessary unit dimensions for visualization
    ax.plot(x_val, y_val, color=color,
            # label=extractor_name,
            linewidth=3)
    ax.fill_between(x_val, y_val - std_val, y_val + std_val, color=color, alpha=0.2)
    if average is not None:
        ax.plot(x_val, average, color='black', alpha=0.3, linestyle='--', label="Feature extractor average")
    if highlighted_residues is not None:
        _plot_highligthed_residues(highlighted_residues, ax)
    ax.set_xlabel("Residue")
    ax.set_ylabel("Importance")
    if set_ylim:
        ax.set_ylim([0, 1.05])

    if show_title:
        ax.set_title(extractor_name)
    else:
        ax.legend()


def _vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel, extractor_name, color, marker, show_legends=False,
                             std_val=None, ylim=None):
    if not (np.isnan(y_val)):
        ax.scatter(x_val, y_val, label=extractor_name, c='w', linewidth=2, marker=marker, s=300, edgecolor='k')
        if ylim is not None:
            ax.set_ylim(ylim)

    if std_val is not None:
        ax.plot([x_val, x_val], [y_val - std_val, y_val + std_val], color='black', alpha=1.0, linewidth=2,
                linestyle='-', marker='s',
                markersize=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legends:
        ax.legend()


def _vis_performance_metrics_box_plot(performance_scores, ax, xlabel, ylabel, extractor_names, colors,
                                      show_legends=False, ylim=None):
    medianprops = dict(color=[0.2, 0.2, 0.2], linewidth=1)
    boxprops = dict(color='k', linewidth=1, facecolor=[0.8, 0.8, 0.8])
    bp = ax.boxplot(performance_scores.T, notch=False, medianprops=medianprops, boxprops=boxprops,
                    labels=extractor_names, patch_artist=True)
    for i_patch, patch in enumerate(bp['boxes']):
        patch.set(facecolor=colors[i_patch], alpha=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    if show_legends:
        ax.legend()


def _vis_per_cluster_projection_entropy(x_val, y_val, width, ax, col, extractor_name, std_val=None, xlabel='',
                                        ylabel='', ylim=None):
    ax.bar(x_val, y_val, width, color=col, edgecolor='', label=extractor_name)
    if std_val is not None:
        for i in range(x_val.shape[0]):
            ax.plot([x_val[i], x_val[i]], [y_val[i] - std_val[i], y_val[i] + std_val[i]], color='black', alpha=0.3,
                    linewidth=1,
                    linestyle='-', marker='s', markersize=1)

    if ylim is not None and not (np.any(np.isnan(ylim))):
        ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return


def _vis_multiple_run_performance_metrics_ave_std(x_vals, metrics, metric_labels, per_cluster_projection_entropies,
                                                  extractor_names, colors, markers):
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

    cluster_proj_entroy_ylim = [0, np.max(
        ave_per_cluster_projection_entropies + std_per_cluster_projection_entropies + 0.1)]

    x_val_clusters = np.arange(ave_per_cluster_projection_entropies.shape[1]) - width * n_estimators / 2.0

    fig1, _ = plt.subplots(1, n_metrics, figsize=(7, 5))
    fig2, _ = plt.subplots(1, 1, figsize=(20, 5))

    for i_metric in range(n_metrics):
        fig1.axes[i_metric].plot(x_vals, ave_metrics[i_metric], color=[0.77, 0.77, 0.82], linewidth=4, zorder=-1)

    x_tick_labels = []

    for i_estimator in range(n_estimators):
        # Visualize each performance metric for current estimator with average+-std, in each axis
        for i_metric in range(n_metrics):
            if i_metric == n_metrics - 1:
                x_tick_labels.append(extractor_names[i_estimator])
            _vis_performance_metrics(x_vals[i_estimator], ave_metrics[i_metric][i_estimator], fig1.axes[i_metric],
                                     'Estimator',
                                     metric_labels[i_metric], extractor_names[i_estimator],
                                     colors[i_estimator % len(colors)], markers[i_estimator],
                                     std_val=std_metrics[i_metric][i_estimator],
                                     show_legends=False, ylim=[0, 1.05])
            if i_estimator == n_estimators - 1:
                fig1.axes[i_metric].xaxis.set_ticks(x_vals)
                fig1.axes[i_metric].set_xticklabels(x_tick_labels)
                fig1.axes[i_metric].set_xlim([x_vals.min() - 0.5, x_vals.max() + 0.5])

        if not (np.any(np.isnan(ave_per_cluster_projection_entropies[i_estimator, :]))):
            _vis_per_cluster_projection_entropy(x_val_clusters + width * i_estimator,
                                                ave_per_cluster_projection_entropies[i_estimator, :], width,
                                                fig2.axes[0],
                                                colors[i_estimator % len(colors)],
                                                extractor_names[i_estimator],
                                                std_val=std_per_cluster_projection_entropies[i_estimator, :],
                                                xlabel='Cluster', ylabel='Projection entropy',
                                                ylim=cluster_proj_entroy_ylim)

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
    n_combi = float(n_dims * (n_dims - 1) / 2)
    counter = 1
    plt.title(title)
    axes = []

    if n_dims == 1:
        plt.scatter(proj_data[:, 0], np.zeros(proj_data.shape[0]), s=15, c=cluster_indices, edgecolor='')
    else:
        plt.axis('off')
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                axes.append(fig.add_subplot(np.ceil(n_combi / 3), 3, counter))
                axes[counter - 1].scatter(proj_data[:, i], proj_data[:, j], s=15, c=cluster_indices, edgecolor='',
                                          alpha=0.3)
                counter += 1
    return


def get_average_feature_importance(postprocessors, i_run):
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
    standard_devs = np.zeros((n_estimators, n_runs))
    test_set_errors = np.zeros((n_estimators, n_runs))
    separation_scores = np.zeros((n_estimators, n_runs))
    projection_entropies = np.zeros((n_estimators, n_runs))
    per_cluster_projection_entropies = np.zeros((n_estimators, n_runs, n_clusters))
    extractor_names = []

    for i_run in range(n_runs):
        for i_estimator in range(n_estimators):
            pp = postprocessors[i_estimator][i_run]
            standard_devs[i_estimator, i_run] = pp.average_std
            test_set_errors[i_estimator, i_run] = pp.test_set_errors
            separation_scores[i_estimator, i_run] = pp.data_projector.separation_score
            projection_entropies[i_estimator, i_run] = pp.data_projector.projection_class_entropy
            per_cluster_projection_entropies[i_estimator, i_run, :] = pp.data_projector.cluster_projection_class_entropy
            if i_run == 0:
                extractor_names.append(pp.extractor.name)

    # metric_labels = ['Average standard deviation', 'Separation score', 'Projection entropy']
    metric_labels = ['Separation score']

    # metrics = [standard_devs, separation_scores, projection_entropies] # Only plot separation scores
    metrics = [separation_scores]

    return x_vals, metrics, metric_labels, per_cluster_projection_entropies, extractor_names


def visualize(postprocessors,
              show_importance=True,
              show_performance=True,
              show_projected_data=False,
              outfile=None,
              highlighted_residues=None,
              show_average=False):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :param show_importance:
    :param show_performance:
    :param show_projected_data:
    :param outfile: if set, save figure to file instead of showing it
    :return:
    """

    n_feature_extractors = len(postprocessors)
    # colors = np.array(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = np.array([_gray]) * n_feature_extractors
    markers = ['o', 's', '>', '^', 'd', 'v', '<']
    if show_performance:
        x_vals, metrics, metric_labels, per_cluster_projection_entropies, extractor_names = extract_metrics(
            postprocessors)

        _vis_multiple_run_performance_metrics_ave_std(x_vals, metrics, metric_labels, per_cluster_projection_entropies,
                                                      extractor_names, colors, markers)

    # Visualize the first run
    i_run = 0
    if show_importance:
        ave_feats, std_feats = get_average_feature_importance(postprocessors, i_run)
        fig1, axes1 = plt.subplots(1, n_feature_extractors, figsize=(6 * n_feature_extractors, 3))
        counter = 0
        for pp, ax in zip(postprocessors, fig1.axes):
            _vis_feature_importance(pp[i_run].get_index_to_resid(), pp[i_run].importance_per_residue,
                                    pp[i_run].std_importance_per_residue,
                                    ax, pp[i_run].extractor.name, colors[counter % len(colors)],
                                    highlighted_residues=highlighted_residues,
                                    average=ave_feats if show_average else None)
            counter += 1

    if show_projected_data:
        fig_counter = 4
        for pp in postprocessors:
            dp = pp[i_run].data_projector
            if dp.projection is not None:
                _vis_projected_data(dp.projection, dp.labels, plt.figure(fig_counter),
                                    "Projection " + pp[i_run].extractor.name)
                fig_counter += 1
    if outfile is None:
        plt.show()
    else:
        plt.tight_layout(pad=0.3)
        plt.savefig(outfile)
        plt.clf()


def _show_performance(postprocessors,
                      xlabels=None,
                      title=None,
                      filename=None,
                      supervised=False,
                      output_dir=None,
                      accuracy_method=None):
    if len(postprocessors) == 0:
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nrows = 2 if supervised else 1
    fig, axs = plt.subplots(nrows, 1, sharex=True, sharey=False, squeeze=False,
                            figsize=(int(0.5 * postprocessors.shape[0]), 2 * nrows))
    accuracy = utils.to_accuracy(postprocessors)
    ax0 = axs[0, 0]
    if title is not None:
        ax0.set_title(title)
    ax0.boxplot(accuracy,
                showmeans=True,
                labels=xlabels,
                patch_artist=True,
                boxprops=_boxprops)
    accuracy_label = "Accuracy"
    if accuracy_method == 'mse':
        accuracy_label = "Accuracy:\nfinding all"
    elif accuracy_method == 'relevant_fraction':
        accuracy_label = "Accuracy:\nignoring irrelevant"
    ax0.set_ylabel(accuracy_label)
    if supervised:
        # Per state
        ax0.get_shared_y_axes().join(ax0, axs[1, 0])  # share y range with regular accuracy
        axs[1, 0].boxplot(utils.to_accuracy_per_cluster(postprocessors),
                          showmeans=True,
                          labels=xlabels,
                          patch_artist=True,
                          boxprops=_boxprops)
        axs[1, 0].set_ylabel("Accuracy\nper state")
        # # Separation score
        # ax2 = axs[2, 0]
        # ax2.boxplot(utils.to_separation_score(postprocessors),
        #             showmeans=True,
        #             labels=xlabels,
        #             patch_artist=True,
        #             boxprops=_boxprops)
        # ax2.set_ylabel("Separation score")

    for [ax] in axs:
        ax.set_xticklabels(xlabels, rotation=45, ha='right')

    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir + filename)
    plt.clf()


def show_single_extractor_performance(postprocessors,
                                      extractor_type,
                                      filename="all.svg",
                                      output_dir="output/benchmarking/",
                                      accuracy_method=None):
    output_dir = "{}/{}/".format(output_dir, extractor_type)
    xlabels = [utils.strip_name(pp.extractor.name) for pp in postprocessors[0]]
    _show_performance(postprocessors,
                      xlabels=xlabels,
                      title=extractor_type,
                      filename=filename,
                      supervised=postprocessors[0, 0].extractor.supervised,
                      output_dir=output_dir,
                      accuracy_method=accuracy_method)


def show_all_extractors_performance(postprocessors,
                                    extractor_types,
                                    feature_type=None,
                                    filename="all.svg",
                                    output_dir="output/benchmarking/",
                                    accuracy_method=None
                                    ):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    xlabels = extractor_types
    supervised = True
    for pp in postprocessors.flatten():
        if not pp.extractor.supervised:
            supervised = False
            break
    title = "Cartesian" if "cart" in feature_type else "Inverse distance"
    _show_performance(postprocessors.T,
                      xlabels=xlabels,
                      title=title,
                      filename=filename,
                      supervised=supervised,
                      output_dir=output_dir,
                      accuracy_method=accuracy_method)
