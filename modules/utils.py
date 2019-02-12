from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.spatial.distance import squareform
from biopandas.pdb import PandasPdb
from sklearn.preprocessing import MinMaxScaler
logger = logging.getLogger("utils")

def vectorize(data):
    """
    Vectorizes the input
    """
    if (len(data.shape)) == 3 and (data.shape[1] == data.shape[2]):
        data_vect = []
        for i in range(data.shape[0]):
            data_vect.append(squareform(data[i, :, :]))
        data_vect = np.asarray(data_vect)
    elif (len(data.shape)) == 2:
        data_vect = data
    else:
        raise Exception("The input array has wrong dimensionality")
    return data_vect


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


def scale(data, remove_outliers=True):
    """
    Scales the input and removes outliers
    """
    perc_2 = np.zeros(data.shape[1])
    perc_98 = np.zeros(data.shape[1])

    #TODO you should be able to turn off this outlier filtration -> now you cannot do inverse scaling!
    if remove_outliers:
        for i in range(data.shape[1]):
            perc_2[i] = np.percentile(data[:, i], 2)
            perc_98[i] = np.percentile(data[:, i], 98)

        for i in range(data.shape[1]):
            perc_2_ind = np.where(data[:, i] < perc_2[i])[0]
            perc_98_ind = np.where(data[:, i] > perc_98[i])[0]
            data[perc_2_ind, i] = perc_2[i]
            data[perc_98_ind, i] = perc_98[i]

    scaler = MinMaxScaler()
    scaler.fit(data)

    data_scaled = scaler.transform(data)

    return data_scaled


def create_class_labels(clustering):
    """
    Transforms a vector of cluster indices to a matrix where a 1 on the ij element means that the ith frame was in cluster state j+1
    """
    number_of_clusters = len(set([t for t in clustering]))
    T = np.zeros((len(clustering), number_of_clusters), dtype=int)
    for i in range(0, len(clustering)):
        T[i, int(clustering[i] - 1)] = 1
    return T


def check_for_overfit(data_scaled, clustering_prob, classifier):
    """
    Checks if the classifier is overfitted
    Computes an error in the form: sum(1-Ptrue+sum(Pfalse))/N_clusters
    """

    clustering_predicted = classifier.predict(data_scaled)

    # Calculate the error as sum(1-Ptrue+sum(Pfalse))/N_clusters
    error_per_frame = np.zeros((clustering_prob.shape[0]))
    number_of_clusters = clustering_prob.shape[1]

    for i in range(clustering_prob.shape[0]):
        error_per_frame[i] = (1 - np.dot(clustering_prob[i], clustering_predicted[i]) + \
                              np.dot(1 - clustering_prob[i], clustering_predicted[i])) / \
                             number_of_clusters
    error = np.average(error_per_frame) * 100
    return error


def rescale_feature_importance(relevances, std_relevances):
    """
    Min-max rescale feature importances
    :param feature_importance: array of dimension nfeatures * nstates
    :param std_feature_importance: array of dimension nfeatures * nstates
    :return: rescaled versions of the inputs with values between 0 and 1
    """

    logger.info("Rescaling feature importances ...")
    if len(relevances.shape) == 1:
        relevances = relevances[:,np.newaxis]
        std_relevances = std_relevances[:,np.newaxis]
    n_states = relevances.shape[1]
    n_features = relevances.shape[0]

    # indices of residues pairs which were not filtered during features filtering
    indices_not_filtered = np.where(relevances[:,0]>=0)[0]

    for i in range(n_states):
        max_val, min_val = relevances[indices_not_filtered,i].max(), relevances[indices_not_filtered,i].min()
        scale = max_val-min_val
        offset = min_val
        if scale < 1e-9:
            scale = max(scale,1e-9)
        relevances[indices_not_filtered,i] = (relevances[indices_not_filtered,i] - offset)/scale
        std_relevances[indices_not_filtered, i] /= scale

    return relevances, std_relevances


def get_default_feature_to_resids(n_features):
    n_residues = 0.5 * (1 + np.sqrt(8 * n_features + 1))
    n_residues = int(n_residues)
    idx = 0
    feature_to_resids = np.empty((n_features, 2))
    for res1 in range(n_residues):
        for res2 in range(res1 + 1, n_residues):
            feature_to_resids[idx, 0] = res1
            feature_to_resids[idx, 1] = res2
            idx += 1
    return feature_to_resids


def get_feature_to_resids_from_pdb(n_features,pdb_file):
    pdb = PandasPdb()
    pdb.read_pdb(pdb_file)
    
    resid_numbers = np.unique(np.asarray(list(pdb.df['ATOM']['residue_number'])))
    n_residues = len(resid_numbers)

    n_residues_check = 0.5 * (1 + np.sqrt(8 * n_features + 1))
    if n_residues!=n_residues_check:
        sys.exit("The number of residues in pdb file ("+str(n_residues)+") is incompatible with number of features ("+str(n_residues_check)+")")

    idx = 0
    feature_to_resids = np.empty((n_features, 2))
    for res1 in range(n_residues):
        for res2 in range(res1 + 1, n_residues):
            feature_to_resids[idx, 0] = resid_numbers[res1]
            feature_to_resids[idx, 1] = resid_numbers[res2]
            idx += 1
    return feature_to_resids
