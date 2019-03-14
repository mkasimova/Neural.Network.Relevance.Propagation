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


def run(nclusters, n_iterations=1, n_splits=1, filetype="pdf", supervised=True, filter_by_distance_cutoff=True):
    feature_type = "ca_inv"  # "contacts_5_cutoff", "closest-heavy_inv" or "CA_inv", "cartesian_ca", "cartesian_noh"
    working_dir = os.path.expanduser("~/projects/gpcr/mega/Result_Data/beta2-dror/clustering_D09/")
    cluster_dir = "{}/clustering_dt10_{}clusters/".format(working_dir, nclusters)
    results_dir = "{}{}/".format(cluster_dir, feature_type)
    suffix = str(nclusters) + "clusters_" + str(n_iterations) + "iterations_" \
             + ("distance-cutoff_" if filter_by_distance_cutoff else "") \
             + (feature_type + "_supervised" if supervised else "_unsupervised")

    # ## Load data
    #
    # Data should be in an appropriate format and filtered before when we load it here. It does not have to be scaled.
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
        'filter_by_distance_cutoff': filter_by_distance_cutoff,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        # 'upper_bound_distance_cutoff': 1.,
        # 'lower_bound_distance_cutoff': 1.
    }
    supervised_feature_extractors = [
        fe.MlpFeatureExtractor(
            hidden_layer_sizes=[int(min(100, data.shape[1]) / (i + 1)) + 1 for i in range(3)],
            training_max_iter=10000,
            alpha=0.0001,
            activation="relu",
            **kwargs),
        # fe.ElmFeatureExtractor(
        #     activation="relu",
        #     n_nodes=data.shape[1]*2,
        #     alpha=0.1,
        #     **kwargs),
        fe.KLFeatureExtractor(**kwargs),
        fe.RandomForestFeatureExtractor(
            one_vs_rest=False,
            n_estimators=1000,
            **kwargs),
    ]
    unsupervised_feature_extractors = [
        fe.MlpAeFeatureExtractor(
            hidden_layer_sizes=(200, 100, 50, 10, 2, 10, 50, 100, 200,),  # int(data.shape[1]/2),),
            # training_max_iter=10000,
            use_reconstruction_for_lrp=True,
            alpha=0.0001,
            activation="relu",
            **kwargs),
        # fe.RbmFeatureExtractor(n_components=8,
        #                        relevance_method='from_components',
        #                        name='RBM_from_components',
        #                        variance_cutoff='auto',
        #                        **kwargs),
        fe.RbmFeatureExtractor(n_components=8,
                               relevance_method='from_lrp',
                               name='RBM',
                               **kwargs),
        fe.PCAFeatureExtractor(n_components=None,
                               variance_cutoff=101,
                               name='PCA',
                               **kwargs),
        # fe.PCAFeatureExtractor(n_components=None,
        #                        name="PCA_%s" % variance_cutoff,
        #                        variance_cutoff=variance_cutoff,
        #                        **kwargs),
        # fe.PCAFeatureExtractor(n_components=None,
        #                        variance_cutoff='6_components',
        #                        name='PCA_6_comp',
        #                        **kwargs),
    ]
    feature_extractors = supervised_feature_extractors if supervised else unsupervised_feature_extractors
    logger.info("Done. using %s feature extractors", len(feature_extractors))

    # # Run the relevance analysis

    for extractor in feature_extractors:
        logger.info("Computing relevance for extractors %s", extractor.name)
        extractor.extract_features()
    logger.info("Done")

    # # Remap and persist results

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

    # # Visualize results
    visualization.visualize(postprocessors,
                            show_importance=True,
                            show_performance=False,
                            show_projected_data=False,
                            outfile=results_dir + "/importance_per_residue_{suffix}.{filetype}".format(suffix=suffix,
                                                                                                       filetype=filetype))

    visualization.visualize(postprocessors,
                            show_importance=False,
                            show_performance=True,
                            show_projected_data=False,
                            outfile=results_dir + "/performance_{suffix}.{filetype}".format(suffix=suffix,
                                                                                            filetype=filetype))

    visualization.visualize(postprocessors,
                            show_importance=False,
                            show_performance=False,
                            show_projected_data=True,
                            outfile=results_dir + "/projected_data_{suffix}.{filetype}".format(suffix=suffix,
                                                                                               filetype=filetype))
    logger.info("Done. The settings were n_iterations = {n_iterations}, n_splits = {n_splits}."
                "\nFiltering (filter_by_distance_cutoff={filter_by_distance_cutoff})".format(**kwargs))


for nclusters in range(2, 7):
    run(nclusters, n_iterations= 5, n_splits=1, supervised=True, filter_by_distance_cutoff=False)
