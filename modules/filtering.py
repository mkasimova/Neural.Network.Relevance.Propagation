from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

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
        ind_nonzero = np.where(relevances[:, i] > 0)
        global_mean = np.mean(relevances[ind_nonzero, i])
        global_sigma = np.std(relevances[ind_nonzero, i])

        # Identify insignificant features
        ind_below_sigma = np.where(relevances[:, i] < \
                                   (global_mean + n_sigma_threshold * global_sigma))[0]
        # Remove insignificant features
        relevances[ind_below_sigma, i] = 0
        std_relevances[ind_below_sigma, i] = 0
    return relevances, std_relevances


def filter_by_contact_cutoff(data, cutoff=0.5):
    """
    Contact cutoff based filtering
    """

    if len(data.shape) != 2:
        raise Exception("The input array has wrong dimensionality; Data should be vectorized first")

    number_of_features = data.shape[1]
    logger.info("Number of features before contact cutoff based filtering is %s", number_of_features)

    data = 1 / data

    data_filtered_ind = []
    for i in range(data.shape[1]):
        data_min = np.min(data[:, i])
        if data_min <= cutoff:
            data_filtered_ind.append(i)

    logger.info("Number of features after contact cutoff based filtering is %s", len(data_filtered_ind))

    data_filtered = data[:, data_filtered_ind]

    data_filtered = 1 / data_filtered

    return data_filtered_ind, data_filtered


def filter_by_DKL(data, clustering, sigma=2, contacts=False):
    """
    DKL based filtering
    """

    if len(data.shape) != 2:
        raise Exception("The input array has wrong dimensionality; Data should be vectorized first")

    number_of_features = data.shape[1]
    logger.info("Number of features before DKL based filtering is %s", number_of_features)

    DKL = np.zeros(number_of_features)

    if contacts == True:

        for i in range(number_of_features):
            var_uniq = list(set(data[:, i]))
            if len(var_uniq) > 1:
                DKL[i] = 1
        data_filtered_ind = np.where(DKL > 0)[0]
        logger.info("Number of features after DKL based filtering is %s", len(data_filtered_ind))
        data_filtered = data[:, data_filtered_ind]

    else:

        std = np.zeros(number_of_features)
        for i in range(number_of_features):
            std[i] = np.std(data[:, i])

        bin_size = np.mean(std)
        logger.info("Bin size for probability calculation is %s", bin_size)

        clustering_var = list(set(clustering))
        ind_cluster_0 = np.where(clustering == clustering_var[0])[0]
        ind_cluster_1 = np.where(clustering == clustering_var[1])[0]
        for i in range(number_of_features):
            DKL[i] = KL_divergence(data[ind_cluster_0, i], data[ind_cluster_1, i], bin_size)

        data_filtered_ind = np.where(DKL >= (np.mean(DKL) + sigma * np.std(DKL)))[0]
        logger.info("Number of features after DKL based filtering is %s", len(data_filtered_ind))
        data_filtered = data[:, data_filtered_ind]

    return DKL, data_filtered_ind, data_filtered


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
