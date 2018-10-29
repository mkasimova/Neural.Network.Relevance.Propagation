from __future__ import absolute_import, division, print_function
import sys
import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import numpy as np
import matplotlib.pyplot as plt
from biopandas.pdb import PandasPdb
import pandas as pd
import modules.utils as utils

logger = logging.getLogger("postprocessing")

def average_and_persist(extractor, relevance_avg, relevance_std, cluster_indices, working_dir, visualize=True,
                        index_to_residue_mapping=None):
    relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue = compute_average_relevance(
        extractor, relevance_avg,
        cluster_indices)
    if index_to_residue_mapping is None:
        index_to_residue_mapping = [resSeq + 1 for resSeq in range(relevance_per_residue.shape[0])]
    persist(extractor, working_dir, relevance_avg, relevance_per_residue_and_cluster, relevance_per_residue, index_to_residue_mapping)
    if visualize:
        plt.plot(index_to_residue_mapping, relevance_per_residue, label=extractor.name)
        plt.xlabel("Residue")
        plt.ylabel("Relevance")
        plt.legend()
        
    return relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue

def compute_relevance_per_residue_and_cluster(relevance):
    logger.warn("Note that we should filter away small relevances here. Annie has code")
    nclusters = 0 if len(relevance.shape) < 2 else relevance.shape[1] 
    if nclusters < 2:
        logger.debug("Not possible to compute relevance per cluster")
        
    n_features = relevance.shape[0]
    n_residues = 0.5*(1+np.sqrt(8*n_features + 1))
    n_residues = int(n_residues)
    relevance_per_residue_and_cluster = np.zeros((n_residues, nclusters))
    feature_idx = 0
    for res1 in range(n_residues):
        for res2 in range(res1 +1, n_residues):
            rel = relevance[feature_idx]
            relevance_per_residue_and_cluster[res1,:] += rel
            relevance_per_residue_and_cluster[res2,:] += rel
            feature_idx += 1
    return relevance_per_residue_and_cluster

def compute_relevance_per_residue(relevance):
    if len(relevance.shape) < 2 or relevance.shape[1] < 2:
        return relevance
    return relevance.mean(axis=1)
    
def compute_average_relevance(extrator, relevance, cluster_indices):
    relevance_per_cluster = relevance # compute_relevance_per_cluster(relevance, cluster_indices)
    relevance_per_residue_and_cluster = compute_relevance_per_residue_and_cluster(relevance_per_cluster)
    relevance_per_residue = compute_relevance_per_residue(relevance_per_residue_and_cluster)
    return relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue


def persist(extractor, working_dir, relevance_per_cluster, relevance_per_residue_and_cluster, relevance_per_residue,
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
    atom = pdb.df['ATOM']
    count = 0
    for i, line in atom.iterrows():
        resSeq = int(line['residue_number'])
        relevance = residue_to_relevance.get(resSeq, None)
        if relevance is None:
            logger.warn("Relevance is None for residue %s", resSeq)
            continue
        atom.set_value(i, 'b_factor', relevance)
    pdb.to_pdb(path=out_file, records=None, gz=False, append_newline=True)


def normalize(values):
    max_val = values.max()
    min_val = values.min()
    scale = max_val - min_val
    offset = min_val
    return (values - offset) / max(scale, 1e-10)
