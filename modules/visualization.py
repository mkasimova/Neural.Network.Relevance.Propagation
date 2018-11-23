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


def vis_performance_metrics(x_val, y_val, ax, xlabel, ylabel, extractor_name, color, show_legends=False,std_val=None):

    if std_val is not None:
        ax.plot([x_val,x_val], [y_val-std_val,y_val+std_val], color='k', linewidth=1, linestyle='--')
    ax.plot(x_val, y_val, label=extractor_name, color=color, linewidth=2, marker='o',markersize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_legends:
        ax.legend()

def vis_multiple_run_performance_metrics(x_vals, standard_devs, entropies, test_set_errors,
                                         separation_scores, projection_entropies, extractor_names, colors):

    n_estimators = standard_devs.shape[0]

    ave_standard_devs = np.mean(standard_devs,axis=1)
    std_standard_devs = np.std(standard_devs, axis=1)

    ave_test_set_errors = np.mean(test_set_errors,axis=1)
    std_test_set_errors = np.std(test_set_errors, axis=1)

    ave_entropies = np.mean(entropies,axis=1)
    std_entropies = np.std(entropies, axis=1)

    ave_separation_scores = np.mean(separation_scores,axis=1)
    std_separation_scores = np.std(separation_scores, axis=1)

    ave_projection_entropies = np.mean(projection_entropies,axis=1)
    std_projection_entropies = np.std(projection_entropies, axis=1)

    for i_estimator in range(n_estimators):
        # Visualize each performance metric for current estimator with average+-std, in each axis

        vis_performance_metrics(x_vals, ave_standard_devs[i_estimator], ax, 'Estimator', 'Average standard deviation',
                                extractor_names[i_estimator], colors[i_estimator],std_val=std_standard_devs)
        vis_performance_metrics(x_vals, ave_test_set_errors[i_estimator], ax, 'Estimator', 'Average test set error',
                                extractor_names[i_estimator], colors[i_estimator],std_val=std_test_set_errors)
        vis_performance_metrics(x_vals, ave_entropies[i_estimator], ax, 'Estimator', 'Relevance entropy',
                                extractor_names[i_estimator], colors[i_estimator],std_val=std_entropies)
        vis_performance_metrics(x_vals, ave_separation_scores[i_estimator], ax, 'Estimator', 'Separation score',
                                extractor_names[i_estimator], colors[i_estimator],std_val=std_separation_scores)
        vis_performance_metrics(x_vals, ave_projection_entropies[i_estimator], ax, 'Estimator', 'Projection entropy',
                                extractor_names[i_estimator], colors[i_estimator],std_val=std_projection_entropies)
    return

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

def get_average_feature_importance(postprocessors,i_run):
    importances = []
    for pp in postprocessors:
        importances.append(pp[i_run].importance_per_residue)
    importances = np.asarray(importances).mean(axis=0)
    importances,_ = pop.rescale_feature_importance(importances, importances)
    return importances

def extract_metrics(postprocessors, data_projection):
    """
    Extract performance metrics from multiple runs.
    :param postprocessors:
    :param data_projection:
    :return:
    """
    n_runs = len(postprocessors[0])
    n_estimators = len(postprocessors)

    x_vals = np.arange(n_estimators)
    standard_devs = np.zeros((n_estimators,n_runs))
    entropies = np.zeros((n_estimators,n_runs))
    test_set_errors = np.zeros((n_estimators,n_runs))
    separation_scores = np.zeros((n_estimators,n_runs))
    projection_entropies = np.zeros((n_estimators,n_runs))
    extractor_names = []

    for i_run in range(n_runs):
        for i_estimator in range(n_estimators):
            standard_devs[i_estimator,i_run] = postprocessors[i_estimator][i_run].average_std
            entropies[i_estimator,i_run] = postprocessors[i_estimator][i_run].entropy
            test_set_errors[i_estimator,i_run] = postprocessors[i_estimator][i_run].test_set_errors
            separation_scores[i_estimator,i_run] = data_projection[i_estimator][i_run].separation_score
            projection_entropies[i_estimator,i_run] = data_projection[i_estimator][i_run].projection_class_entropy
            if i_run == 0:
                extractor_names.append(postprocessors[i_estimator][i_run].extractor.name)
    return x_vals, standard_devs, entropies, test_set_errors, separation_scores, projection_entropies, extractor_names

def visualize(postprocessors, data_projectors, show_importance=True, show_performance=True, show_projected_data=False):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    n_feature_extractors = len(postprocessors)
    cols = np.asarray([[0.7, 0.0, 0.08], [0, 0.5, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.9, 0.6, 0.2], [0, 0, 0]])

    if show_performance:
        if postprocessors[0][0].predefined_relevant_residues is None:
            fig2, axes2 = plt.subplots(1, 4, figsize=(32, 3))
        else:
            fig2, axes2 = plt.subplots(1, 6, figsize=(45, 3))

        x_vals, standard_devs, entropies, test_set_errors, separation_scores, projection_entropies, extractor_names = \
            extract_metrics(postprocessors, data_projectors)

        vis_multiple_run_performance_metrics(x_vals, standard_devs, entropies,
                                         separation_scores, projection_entropies, test_set_errors, extractor_names, cols)

    # Visualize the first run
    i_run = 0
    if show_importance:
        ave_feats = get_average_feature_importance(postprocessors, i_run)

        fig1, axes1 = plt.subplots(1, n_feature_extractors, figsize=(35, 3))
        for pp, ax in zip(postprocessors, fig1.axes):
            vis_feature_importance(pp[i_run].index_to_resid, pp[i_run].importance_per_residue, pp[i_run].std_importance_per_residue,
                                   ax, pp[i_run].extractor.name, cols[counter, :], average=ave_feats)

    """if show_performance:   
        x_val_prev = None
        sep_score_prev = None
        std_prev = None
        entropy_prev= None
        proj_entropy_prev = None
        
        counter = 0
        for pp, dp in zip(postprocessors, data_projectors):
    
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

                vis_performance_metrics(counter, dp.projection_class_entropy, fig2.axes[3], 'Estimator',
                                        'Projection posterior entropy',pp.extractor.name, cols[counter, :],
                                        show_legends=True, x_val_prev=x_val_prev,y_val_prev=proj_entropy_prev)
            
            x_val_prev = counter
            std_prev = pp.average_std
            entropy_prev = pp.entropy
            sep_score_prev = dp.separation_score
            proj_entropy_prev = dp.projection_class_entropy
            counter+=1
    """

    if show_projected_data:
        fig_counter = 3
        for pp, dp in zip(postprocessors, data_projectors):
            if dp.raw_projection is not None:
                vis_projected_data(dp[i_run].raw_projection, dp[i_run].labels, plt.figure(fig_counter), "Raw projection "+dp[i_run].extractor.name)
                fig_counter += 1


    plt.show()
