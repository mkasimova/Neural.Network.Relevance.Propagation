from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
import pandas as pd
import modules.utils as utils
from scipy.spatial.distance import squareform


def compute_relevance_per_cluster(all_relevances, class_indices):
    """
	Average relevance over each state/cluster. 
	"""
    if len(all_relevances) == 3:
        class_labels = np.unique(class_indices)
        n_classes = class_labels.shape[0]

        class_relevance = np.zeros((all_relevances.shape[0], n_classes, all_relevances.shape[2]))

        for k in range(n_classes):
            class_relevance[:, k, :] = \
                np.mean(all_relevances[:, np.where(clustering_train == \
                                                   class_labels[k])[0], :], axis=0)

        return class_relevance
    else:
        return all_relevances


def rescale_feature_importance(feature_importance):
    """
	Min-max rescale feature importances
	"""
    if len(feature_importance.shape) == 3:
        for i in range(feature_importance.shape[0]):
            for j in range(feature_importance.shape[1]):
                feature_importance[i, j, :] = (feature_importance[i, j, :] - np.min(feature_importance[i, j, :])) / \
                                              (np.max(feature_importance[i, j, :]) - np.min(
                                                  feature_importance[i, j, :]) + 1e-9)

    return feature_importance


def residue_importances(feature_importances):
    """
	Compute residue importance
	"""
    if len(feature_importances.shape) == 1:
        n_states = 1
        feature_importances = feature_importances[:, np.newaxis].T
    else:
        n_states = feature_importances.shape[0]

    n_residues = squareform(feature_importances[0, :]).shape[0]

    resid_importance = np.zeros((n_states, n_residues))
    print(resid_importance.shape)
    for i_state in range(n_states):
        resid_importance[i_state, :] = np.sum(squareform(feature_importances[i_state, :]), axis=1)
    return resid_importance


def average_and_persist(extractor, relevance, cluster_indices, working_dir, visualize=True,
                        index_to_residue_mapping=None):
    relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue = compute_average_relevance(
        extractor, relevance,
        cluster_indices)
    if index_to_residue_mapping is None:
        index_to_residue_mapping = [resSeq + 1 for resSeq in range(relevance_per_residue.shape[0])]
    persist(working_dir, relevance_per_residue_and_cluster, relevance_per_residue, index_to_residue_mapping)
    if visualize:
        plt.plot(index_to_residue_mapping, relevance_per_residue, label=extractor.name)
        plt.xlabel("Residue")
        plt.ylabel("Relevance")
        plt.legend()

    return relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue


def compute_average_relevance(extrator, relevance, cluster_indices):
    relevance_per_cluster = compute_relevance_per_cluster(relevance, cluster_indices)
    relevance_per_residue_and_cluster = compute_relevance_per_residue(relevance_per_cluster)
    relevance_per_residue = compute_relevance_per_residue(relevance_per_residue_and_cluster)
    return relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue


def persist(working_dir, relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue,
            index_to_residue_mapping):
    directory = working_dir + "analysis/{}/".format(extractor.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.save(directory + "relevance_per_cluster", relevance_per_cluster)
    np.save(directory + "relevance_per_residue_and_cluster", relevance_per_residue_and_cluster)
    np.save(directory + "relevance_per_residue", relevance_per_residue)

    pdb_file = working_dir + "analysis/all.pdb"  # TODO don't hard code
    pdb = PandasPdb()
    pdb.read_pdb(pdb_file)
    save_to_pdb(pdb, directory + "all_relevance.pdb",
                map_to_correct_residues(relevance_per_residue, index_to_residue_mapping))
    for cluster_idx, relevance in enumerate(relevance_per_residue_and_cluster):
        save_to_pdb(pdb, directory + "cluster_{}_relevance.pdb".format(cluster_idx),
                    map_to_correct_residues(relevance, index_to_residue_mapping))


def map_to_correct_residues(relevance_per_residue, index_to_residue_mapping):
    residue_to_relevance = {}
    for idx, rel in enumerate(relevance_per_residue):
        resSeq = index_to_residue_mapping[idx]
        residue_to_relevance[resSeq] = rel
    return residue_to_relevance


def save_to_pdb(pdb, out_file, residue_to_relevance):
    beta = 'results.dat'

    dl = [data.rstrip('\n') for data in open(rdir + beta)]
    chain = len(dl)

    atom = ppdb.df['ATOM']
    count = 0

    for i, line in atom.iterrows():
        atom.set_value(i, 'b_factor', float(dl[count].split()[0]) / 10.)
        # print(dl[count].split()[0])
        # print(line['residue_number'])
        try:
            if line['residue_number'] != atom.loc[i + 1, 'residue_number']:
                count = (count + 1) % chain
            # else:
            # print("NOOOO", line['residue_number'])
        except KeyError:
            pass
    ppdb.to_pdb(path=rdir + pdbfid + '.beta.pdb', records=None, gz=False, append_newline=True)


def normalize(values):
    max_val = values.max()
    min_val = values.min()
    scale = max_val - min_val
    offset = min_val
    return (values - offset) / max(scale, 1e-10)
