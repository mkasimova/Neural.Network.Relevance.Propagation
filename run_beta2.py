# coding: utf-8

# # Init

# In[1]:


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
from modules import utils, feature_extraction as fe, postprocessing, visualization

logger = logging.getLogger("beta2")
utils.remove_outliers = False


# ## Load data
# 
# Data should be in an appropriate format and filtered before when we load it here. It does not have to be scaled. 

# In[9]:

def run(nclusters, n_iterations=1, n_splits=1):
    feature_type = "ca_inv"  # "contacts_5_cutoff", "closest-heavy_inv" or "CA_inv", "cartesian_ca", "cartesian_noh"
    working_dir = os.path.expanduser("~/projects/gpcr/mega/Result_Data/beta2-dror/clustering_D09/")
    cluster_dir = "{}/clustering_dt10_{}clusters/".format(working_dir, nclusters)
    results_dir = "{}{}/".format(cluster_dir, feature_type)

    data = np.load("{}/samples/{}/samples.npz".format(working_dir, feature_type))['array']
    feature_to_resids = np.load("{}/samples/{}/feature_to_resids.npy".format(working_dir, feature_type))
    cluster_indices = np.loadtxt(cluster_dir + "cluster_indices_.txt")
    cluster_indices -= 1  # start at 0 instead of 1
    if len(data) != len(cluster_indices) or data.shape[1] != len(feature_to_resids):
        raise Exception()
    logger.info("Loaded data of shape %s and %s clusters for %s clusters and feature type %s", data.shape,
                len(set(cluster_indices)), nclusters, feature_type)

    # ## Define the different methods to use
    #
    # Every method is encapsulated in a so called FeatureExtractor class which all follow the same interface

    # rbm_data = np.copy(data)
    # np.random.shuffle(rbm_data)
    kwargs = {
        'samples': data,
        'cluster_indices': cluster_indices,
        'filter_by_distance_cutoff': True,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        # 'upper_bound_distance_cutoff': 1.,
        # 'lower_bound_distance_cutoff': 1.
    }
    feature_extractors = [
        # fe.MlpFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
        #                        hidden_layer_sizes=(100,),  # , 50, 25),
        #                        activation="logistic",
        #                        use_inverse_distances=use_inverse_distances,
        #                        randomize=True,
        #                        filter_by_distance_cutoff=filter_by_distance_cutoff),
        # fe.MlpAeFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
        #                          hidden_layer_sizes=(200, 2, 200),
        #                          training_max_iter=200,
        #                          use_inverse_distances=use_inverse_distances,
        #                          activation="logistic"),  # , solver="sgd"),
        # fe.RbmFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
        #                        n_components=nclusters,
        #                        use_inverse_distances=use_inverse_distances,
        #                        filter_by_distance_cutoff=filter_by_distance_cutoff),
        # fe.ElmFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
        #                        filter_by_distance_cutoff=filter_by_distance_cutoff,
        #                        n_nodes=1000,
        #                        use_inverse_distances=use_inverse_distances,
        #                        alpha=1000,
        #                        activation="relu"),
        fe.KLFeatureExtractor(**kwargs),
        # fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits,
        #                        use_inverse_distances=use_inverse_distances,
        #                        filter_by_distance_cutoff=filter_by_distance_cutoff),
        # fe.RandomForestFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
        #                                 use_inverse_distances=use_inverse_distances,
        #                                 filter_by_distance_cutoff=filter_by_distance_cutoff),
    ]
    logger.info("Done. using %s feature extractors", len(feature_extractors))

    # # Run the relevance analysis

    # In[14]:

    for extractor in feature_extractors:
        logger.info("Computing relevance for extractors %s", extractor.name)
        extractor.extract_features()
    logger.info("Done")

    # # Remap and persist results

    # In[15]:

    postprocessors = []
    for extractor in feature_extractors:
        p = postprocessing.PostProcessor(extractor,
                                         working_dir=results_dir,
                                         pdb_file=working_dir + "/trajectories/protein_noh.pdb",
                                         feature_to_resids=feature_to_resids,
                                         filter_results=True)
        p.average()
        p.evaluate_performance()
        p.persist()
        postprocessors.append([p])
    logger.info("Done")

    # # Visualize results3

    visualization.visualize(postprocessors,
                            show_importance=True,
                            show_performance=False,
                            show_projected_data=False,
                            outfile=results_dir + "/importance_per_residue.svg")

    visualization.visualize(postprocessors,
                            show_importance=False,
                            show_performance=True,
                            show_projected_data=False,
                            outfile=results_dir + "/performance.svg")

    visualization.visualize(postprocessors,
                            show_importance=False,
                            show_performance=False,
                            show_projected_data=True,
                            outfile=results_dir + "/projected_data.svg")
    logger.info("Done. The settings were n_iterations = {n_iterations}, n_splits = {n_splits}."
                "\nFiltering (filter_by_distance_cutoff={filter_by_distance_cutoff})".format(**kwargs))


for nclusters in range(2, 7):
    run(nclusters, n_iterations=1, n_splits=1)
