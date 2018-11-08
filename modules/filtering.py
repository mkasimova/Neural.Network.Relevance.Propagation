from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.stats import entropy
from scipy.stats import ks_2samp as ks_test

logger = logging.getLogger("filtering")


def filter_feature_importance(relevances, std_relevances, n_sigma_threshold=2):
    """
    Filter feature importances based on significance.
    Return filtered residue feature importances (average + std within the states/clusters).
    """
    if len(relevances.shape) == 1:
        n_states = 1
        relevances = relevances[:, np.newaxis]
        std_relevances = std_relevances[:,np.newaxis]
    else:
        n_states = relevances.shape[1]

    n_features = relevances.shape[0]

    for i in range(n_states):
        ind_nonzero = np.where(relevances[:, i] > 0) # TODO missing [0]?
        global_mean = np.mean(relevances[ind_nonzero, i])
        global_sigma = np.std(relevances[ind_nonzero, i])

        # Identify insignificant features
        ind_below_sigma = np.where(relevances[:, i] < (global_mean + n_sigma_threshold * global_sigma))[0]
        # Remove insignificant features
        relevances[ind_below_sigma, i] = 0
        std_relevances[ind_below_sigma, i] = 0
    return relevances, std_relevances


def filter_by_distance_cutoff(data, indices_for_filtering, cutoff=0.5, inverse_distances=True):
    """
    Contact cutoff based filtering
    """

    number_of_features = data.shape[1]
    logger.info("Number of features before distance cutoff based filtering is %s", number_of_features)

    if inverse_distances:
        data = 1 / data

    data_filtered_ind = []
    for i in range(data.shape[1]):
        data_min = np.min(data[:, i])
        if data_min <= cutoff:
            data_filtered_ind.append(i)

    logger.info("Number of features after distance cutoff based filtering is %s", len(data_filtered_ind))

    data_filtered = data[:, data_filtered_ind]
    indices_for_filtering = indices_for_filtering[data_filtered_ind]

    if inverse_distances:
        data_filtered = 1 / data_filtered

    return data_filtered, indices_for_filtering


def KL_divergence(x,y,bin_size):
    """
    Compute Kullback-Leibler divergence
    """
    bin_min = np.min(np.concatenate((x,y)))
    bin_max = np.max(np.concatenate((x,y)))
    if bin_size>=(bin_max-bin_min):
        DKL=0
    else:
        bin_n = int((bin_max-bin_min)/bin_size)
        x_prob = np.histogram(x,bins=bin_n,range=(bin_min,bin_max),density=True)[0]+0.000000001
        y_prob = np.histogram(y,bins=bin_n,range=(bin_min,bin_max),density=True)[0]+0.000000001
        DKL = 0.5*(entropy(x_prob,y_prob)+entropy(y_prob,x_prob))
    return DKL


def filter_by_KS_test(data, clustering, indices_for_filtering, p_value_threshold=0.0001):
    """
    KS test based filtering; adjust p_value_threshold if needed
    """

    number_of_features = data.shape[1]
    logger.info("Number of features before KS test based filtering is %s", number_of_features)

    clustering_var = list(set(clustering))
    n_clusters = len(clustering_var)

    '''
    data_filtered_ind = []
    for i in range(number_of_features):
        for j in range(n_clusters):
            # cluster 0 - selected cluster; cluster 1 - all the other clusters except cluster 0
            ind_cluster_0 = np.where(clustering == clustering_var[j])[0]
            ind_cluster_1 = np.where(clustering != clustering_var[j])[0]
            statistics, p_value = ks_test(data[ind_cluster_0, i], data[ind_cluster_1, i])
            if p_value<=p_value_threshold:
                data_filtered_ind.append(i)
                break
    '''

    p_value_total = np.zeros(number_of_features)

    for i in range(n_clusters):
        # cluster 0 - selected cluster; cluster 1 - all the other clusters except cluster 0
        ind_cluster_0 = np.where(clustering == clustering_var[i])[0]
        ind_cluster_1 = np.where(clustering != clustering_var[i])[0]
        for j in range(number_of_features):
            stats, p_value = ks_test(data[ind_cluster_0, j], data[ind_cluster_1, j])
            p_value_total[j] = p_value_total[j] + p_value

    p_value_total = p_value_total/n_clusters
    data_filtered_ind = np.where(p_value_total<=p_value_threshold)[0]

    logger.info("Number of features after KS test based filtering is %s", len(data_filtered_ind))
    data_filtered = data[:, data_filtered_ind]
    indices_for_filtering = indices_for_filtering[data_filtered_ind]

    return data_filtered, indices_for_filtering


def filter_by_DKL(data, clustering, indices_for_filtering, sigma=2):
    """
    DKL based filtering
    """

    number_of_features = data.shape[1]
    logger.info("Number of features before DKL based filtering is %s", number_of_features)

    DKL = np.zeros(number_of_features)

    std = np.zeros(number_of_features)
    for i in range(number_of_features):
        std[i] = np.std(data[:, i])

    bin_size = np.mean(std)
    logger.info("Bin size for probability calculation is %s", bin_size)

    clustering_var = list(set(clustering))
    n_clusters = len(clustering_var)

    for i in range(n_clusters):
        # cluster 0 - selected cluster; cluster 1 - all the other clusters except cluster 0
        ind_cluster_0 = np.where(clustering == clustering_var[i])[0]
        ind_cluster_1 = np.where(clustering != clustering_var[i])[0]
        for j in range(number_of_features):
            DKL[j] += KL_divergence(data[ind_cluster_0, j], data[ind_cluster_1, j], bin_size)

    DKL = DKL/n_clusters

    data_filtered_ind = np.where(DKL >= (np.mean(DKL) + sigma * np.std(DKL)))[0]
    logger.info("Number of features after DKL based filtering is %s", len(data_filtered_ind))
    data_filtered = data[:, data_filtered_ind]
    indices_for_filtering = indices_for_filtering[data_filtered_ind]

    return data_filtered, indices_for_filtering


def filter_for_data_as_contacts(data, indices_for_filtering):
    """
    Remove all features wich always have either 1 (contact) or 0 (no contact)
    """

    number_of_features = data.shape[1]
    logger.info("Number of features before filtering of contacts is %s", number_of_features)

    contacts_filter = np.zeros(number_of_features)

    for i in range(number_of_features):
        var_uniq = list(set(data[:, i]))
        if len(var_uniq) > 1:
            contacts_filter[i] = 1

    data_filtered_ind = np.where(contacts_filter > 0)[0]
    logger.info("Number of features after filtering of contacts is %s", len(data_filtered_ind))
    data_filtered = data[:, data_filtered_ind]
    indices_for_filtering = indices_for_filtering[data_filtered_ind]

    return data_filtered, indices_for_filtering


def keep_datapoints(data, clustering, points_to_keep=[]):
    """
    Keeps selected datapoints in a sample (used when clustering is not clean)
    """
    if len(points_to_keep) == 0:
        data_keep = data
        clustering_keep = clustering
    else:
        logger.info("Discarding points ...")
        logger.info("Number of points before discarding is %s", data.shape[0])
        points_to_keep = np.asarray(points_to_keep)
        for i in range(len(points_to_keep)):
            if i == 0:
                data_keep = data[points_to_keep[i, 0]:points_to_keep[i, 1]]
                clustering_keep = clustering[points_to_keep[i, 0]:points_to_keep[i, 1]]
            else:
                data_keep = np.vstack((data_keep, data[points_to_keep[i, 0]:points_to_keep[i, 1]]))
                clustering_keep = np.concatenate(
                    (clustering_keep, clustering[points_to_keep[i, 0]:points_to_keep[i, 1]]))
        logger.info("Number of points after discarding is %s", data_keep.shape[0])

    return data_keep, clustering_keep


def remap_after_filtering(feats,std_feats,n_features,res_indices_for_filtering):
    """
    After filtering remaps features to the matrix with initial dimensions
    """
    feats_remapped = np.zeros((n_features,feats.shape[1]))
    feats_remapped[res_indices_for_filtering,:] = feats

    std_feats_remapped = np.zeros((n_features,std_feats.shape[1]))
    std_feats_remapped[res_indices_for_filtering,:] = std_feats

    return feats_remapped, std_feats_remapped

