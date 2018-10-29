from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import numpy as np
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
import modules.utils as utils
from scipy.spatial.distance import squareform

logger = logging.getLogger("postprocessing")


def average_and_persist(extractor, feature_importance, std_feature_importance, cluster_indices, working_dir,
                        visualize=True,
                        feature_to_resids=None):
    (nfeatures, nclusters) = feature_importance.shape
    if feature_to_resids is None:
        feature_to_resids = utils.get_default_feature_to_resids(nfeatures)
    importance_per_cluster, importance_per_residue_and_cluster, importance_per_residue, index_to_resid = compute_average_importance(
        # extractor,
        feature_importance,
        cluster_indices,
        feature_to_resids)
    persist(extractor, working_dir, feature_importance, importance_per_residue_and_cluster, importance_per_residue,
            index_to_resid)
    if visualize:
        # TODO visualize features too with std
        plt.plot(index_to_resid, importance_per_residue, label=extractor.name)
        plt.xlabel("Residue")
        plt.ylabel("importance")
        plt.legend()

    return importance_per_cluster, importance_per_residue_and_cluster, importance_per_residue


def rescale_feature_importance(feature_importance, std_feature_importance):
    """
	Min-max rescale feature importances
	"""
    if len(feature_importance.shape) == 3:
        for i in range(feature_importance.shape[0]):
            for j in range(feature_importance.shape[1]):
                min_X = np.min(feature_importance[i, j, :])
                max_X = np.max(feature_importance[i, j, :])
                std_feature_importance[i, j, :] /= (max_X - min_X + 1e-9)
                feature_importance[i, j, :] = (feature_importance[i, j, :] - min_X) / \
                                              (max_X - min_X + 1e-9)
    return feature_importance, std_feature_importance


def residue_importances(feature_importances, std_feature_importances):
    """
	Compute residue importance
	"""
    if len(feature_importances.shape) == 1:
        n_states = 1
        feature_importances = feature_importances[:, np.newaxis].T
        std_feature_importances = std_feature_importances[:, np.newaxis].T
    else:
        n_states = feature_importances.shape[0]

    n_residues = squareform(feature_importances[0, :]).shape[0]

    resid_importance = np.zeros((n_states, n_residues))
    std_resid_importance = np.zeros((n_states, n_residues))
    for i_state in range(n_states):
        resid_importance[i_state, :] = np.sum(squareform(feature_importances[i_state, :]), axis=1)
        std_resid_importance[i_state, :] = np.sqrt(np.sum(squareform(std_feature_importances[i_state, :] ** 2), axis=1))
    return resid_importance, std_resid_importance


def compute_importance_per_residue_and_cluster(importance, feature_to_resids):
    nclusters = 0 if len(importance.shape) < 2 else importance.shape[1]
    if nclusters < 2:
        logger.debug("Not possible to compute importance per cluster")

    n_features = importance.shape[0]
    index_to_resid = set(feature_to_resids.flatten())  # at index X we have residue number
    index_to_resid = [r for r in index_to_resid]
    res_id_to_index = {}  # a map pointing back to the index in the array index_to_resid
    for idx, resid in enumerate(index_to_resid):
        res_id_to_index[resid] = idx
    n_residues = len(index_to_resid)
    n_residues = int(n_residues)
    importance_per_residue_and_cluster = np.zeros((n_residues, nclusters))
    for feature_idx, rel in enumerate(importance):
        res1, res2 = feature_to_resids[feature_idx]
        res1 = res_id_to_index[res1]
        res2 = res_id_to_index[res2]
        importance_per_residue_and_cluster[res1, :] += rel
        importance_per_residue_and_cluster[res2, :] += rel
    return importance_per_residue_and_cluster, index_to_resid


def compute_importance_per_residue(importance_per_residue_and_cluster):
    if len(importance_per_residue_and_cluster.shape) < 2 or importance_per_residue_and_cluster.shape[1] < 2:
        return importance_per_residue_and_cluster
    return importance_per_residue_and_cluster.mean(axis=1)


def compute_average_importance(feature_importance, cluster_indices, feature_to_resids):
    importance_per_cluster = feature_importance  # compute_importance_per_cluster(importance, cluster_indices)
    importance_per_residue_and_cluster, index_to_resid = compute_importance_per_residue_and_cluster(
        importance_per_cluster, feature_to_resids)
    importance_per_residue = compute_importance_per_residue(importance_per_residue_and_cluster)
    return importance_per_cluster, importance_per_residue_and_cluster, importance_per_residue, index_to_resid


def persist(extractor, working_dir, importance_per_cluster, importance_per_residue_and_cluster, importance_per_residue,
            index_to_resid):
    # TODO don't hard code filepaths below!
    directory = working_dir + "analysis/{}/".format(extractor.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory + "importance_per_cluster", importance_per_cluster)
    np.save(directory + "importance_per_residue_and_cluster", importance_per_residue_and_cluster)
    np.save(directory + "importance_per_residue", importance_per_residue)

    pdb_file = working_dir + "analysis/all.pdb"
    pdb = PandasPdb()
    pdb.read_pdb(pdb_file)
    save_to_pdb(pdb, directory + "all_importance.pdb",
                map_to_correct_residues(importance_per_residue, index_to_resid))
    for cluster_idx, importance in enumerate(importance_per_residue_and_cluster.T):
        save_to_pdb(pdb, directory + "cluster_{}_importance.pdb".format(cluster_idx),
                    map_to_correct_residues(importance, index_to_resid))


def map_to_correct_residues(importance_per_residue, index_to_resid):
    residue_to_importance = {}
    for idx, rel in enumerate(importance_per_residue):
        resSeq = index_to_resid[idx]
        residue_to_importance[resSeq] = rel
    return residue_to_importance


def save_to_pdb(pdb, out_file, residue_to_importance):
    atom = pdb.df['ATOM']
    count = 0
    for i, line in atom.iterrows():
        resSeq = int(line['residue_number'])
        importance = residue_to_importance.get(resSeq, None)
        # print(i, resSeq, importance)
        if importance is None:
            logger.warn("importance is None for residue %s and line %s", resSeq, line)
            continue
        atom.set_value(i, 'b_factor', importance)
    pdb.to_pdb(path=out_file, records=None, gz=False, append_newline=True)


def normalize(values):
    max_val = values.max()
    min_val = values.min()
    scale = max_val - min_val
    offset = min_val
    return (values - offset) / max(scale, 1e-10)


def state_average(self, all_importances, class_indices):
    """
    Average importance over each state/cluster. 
    """
    if len(all_importances) == 3:
        class_labels = np.unique(class_indices)
        n_classes = class_labels.shape[0]

        class_importance = np.zeros((all_importances.shape[0], n_classes, all_importances.shape[2]))

        for k in range(n_classes):
            class_importance[:, k, :] = \
                np.mean(all_importances[:, np.where(clustering_train == \
                                                    class_labels[k])[0], :], axis=0)

        return class_importance
    else:
        return all_importances


def filter_feature_importance(self, average_importance, std_importance=None, n_sigma_threshold=2):
    """
    Filter feature importances based on significance.
    Return filtered residue feature importances (average + std within the states/clusters).
    """
    if len(average_importance.shape) == 1:
        n_states = 1
        n_features = squareform(average_importance[:]).shape[0]
        average_importance = average_importance[:, np.newaxis].T
    else:
        n_states = average_importance.shape[0]
        n_features = squareform(average_importance[0, :]).shape[0]

    residue_importance_ave = np.zeros((n_states, n_features))
    residue_importance_std = np.zeros((n_states, n_features))

    for i in range(n_states):
        ind_nonzero = np.where(average_importance[i, :] > 0)
        global_mean = np.mean(average_importance[i, ind_nonzero])
        global_sigma = np.std(average_importance[i, ind_nonzero])

        average_importance_mat = squareform(average_importance[i, :])

        # If no standard deviation present, 
        if std_importance is not None:
            std_importance_mat = squareform(std_importance[i, :])
        else:
            residue_importance_std = np.zeros(residue_importance_ave.shape)

        for j in range(n_features):
            # Identify significant features         
            ind_above_sigma = np.where(average_importance_mat[j, :] >= \
                                       (global_mean + n_sigma_threshold * global_sigma))[0]

            # Sum over significant features (=> per-residue importance)
            residue_importance_ave[i, j] = np.sum(average_importance_mat[j, ind_above_sigma])
            if std_importance is not None:
                residue_importance_std[i, j] = np.sqrt(np.sum(std_importance_mat[j, ind_above_sigma] ** 2))

    return residue_importance_ave, residue_importance_std
