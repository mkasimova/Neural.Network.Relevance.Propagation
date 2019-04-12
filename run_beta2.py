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
from modules import utils, feature_extraction as fe, visualization

logger = logging.getLogger("beta2")
utils.remove_outliers = False


def run(nclusters=2,
        simu_type="clustering",
        n_iterations=1,
        n_splits=1,
        shuffle_datasets=True,
        dt=10,
        feature_type="ca_inv",  # "contacts_5_cutoff", "closest-heavy_inv" or "CA_inv", "cartesian_ca", "cartesian_noh"
        filetype="svg",
        supervised=True,
        filter_by_distance_cutoff=True):
    if simu_type == "clustering":
        working_dir = os.path.expanduser("~/projects/gpcr/mega/Result_Data/beta2-dror/clustering_D09/")
        cluster_dir = "{}/clustering_dt{}_{}clusters/".format(working_dir, dt, nclusters)
        results_dir = "{}{}/".format(cluster_dir, feature_type)
        data = np.load("{}/samples/{}/samples.npz".format(working_dir, feature_type, dt))['array']
        feature_to_resids = np.load("{}/samples/{}/feature_to_resids.npy".format(working_dir, feature_type))
        cluster_indices = np.loadtxt(cluster_dir + "cluster_indices_.txt")
    elif simu_type == "apo-holo":
        working_dir = os.path.expanduser("~/projects/gpcr/mega/Result_Data/beta2-dror/apo-holo/")
        results_dir = "{}/results/{}/".format(working_dir, feature_type)
        data = np.load("{}/samples/{}/samples_dt{}.npz".format(working_dir, feature_type, dt))['array']
        feature_to_resids = np.load("{}/samples/{}/feature_to_resids.npy".format(working_dir, feature_type))
        cluster_indices = np.loadtxt("{wd}/cluster_indices/cluster_indices_dt{dt}.txt".format(wd=working_dir, dt=dt))
    else:
        raise Exception("Unsupported simulation type {simu_type}".format(simu_type=simu_type))
    supervised_suffix = ""
    if supervised is not None:
        supervised_suffix = "_supervised" if supervised else "_unsupervised"
    suffix = str(nclusters) + "clusters_" + str(n_iterations) + "iterations_" \
             + ("distance-cutoff_" if filter_by_distance_cutoff else "") \
             + (feature_type + supervised_suffix)
    cluster_indices -= 1  # start at 0 instead of 1
    if len(data) != len(cluster_indices) or data.shape[1] != len(feature_to_resids):
        raise Exception("Inconsistent input data. The number of features or the number of frames to no match")
    logger.info("Loaded data of shape %s and %s clusters for %s clusters and feature type %s", data.shape,
                len(set(cluster_indices)), nclusters, feature_type)
    # ## Define the different methods to use
    #
    # Every method is encapsulated in a so called FeatureExtractor class which all follow the same interface
    kwargs = {
        'samples': data,
        'cluster_indices': cluster_indices,
        'filter_by_distance_cutoff': filter_by_distance_cutoff,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        'shuffle_datasets': shuffle_datasets
        # 'upper_bound_distance_cutoff': 1.,
        # 'lower_bound_distance_cutoff': 1.
    }
    unsupervised_feature_extractors = [
        # fe.RbmFeatureExtractor(n_components=8,
        #                        relevance_method='from_components',
        #                        name='RBM_from_components',
        #                        variance_cutoff='auto',
        #                        **kwargs),
        fe.PCAFeatureExtractor(classifier_kwargs={'n_components': None},
                               variance_cutoff=75,
                               name='PCA',
                               **kwargs),
        fe.RbmFeatureExtractor(classifier_kwargs={'n_components': 1},
                               relevance_method='from_lrp',
                               name='RBM',
                               **kwargs),
        # fe.PCAFeatureExtractor(n_components=None,
        #                        name="PCA_%s" % variance_cutoff,
        #                        variance_cutoff=variance_cutoff,
        #                        **kwargs),
        # fe.PCAFeatureExtractor(n_components=None,
        #                        variance_cutoff='6_components',
        #                        name='PCA_6_comp',
        #                        **kwargs),
        fe.MlpAeFeatureExtractor(
            classifier_kwargs={
                'hidden_layer_sizes': (100, 50, 10, 2, 10, 50, 100,),  # int(data.shape[1]/2),),
                # max_iter=10000,
                'alpha': 0.0001,
                'activation': "logistic"
            },
            use_reconstruction_for_lrp=True,
            **kwargs),
    ]
    supervised_feature_extractors = [
        # fe.ElmFeatureExtractor(
        #     activation="relu",
        #     n_nodes=data.shape[1] * 2,
        #     alpha=0.1,
        #     **kwargs),
        # fe.RandomForestFeatureExtractor(
        #     one_vs_rest=False,
        #     classifier_kwargs={'n_estimators': 1000},
        #     **kwargs),
        # fe.KLFeatureExtractor(**kwargs),
        fe.MlpFeatureExtractor(
            classifier_kwargs={
                'hidden_layer_sizes': [int(min(100, data.shape[1]) / (i + 1)) + 1 for i in range(3)],
                'max_iter': 10000,
                'alpha': 0.0001,
                'activation': "relu"
            },
            per_frame_importance_outfile="/home/oliverfl/projects/gpcr/mega/Result_Data/beta2-dror/clustering_D09/trajectories/mlp_perframeimportance.txt",
            **kwargs),
    ]

    if supervised is None:
        feature_extractors = unsupervised_feature_extractors + supervised_feature_extractors
    else:
        feature_extractors = supervised_feature_extractors if supervised else unsupervised_feature_extractors
    logger.info("Done. using %s feature extractors", len(feature_extractors))

    # # Run the relevance analysis
    postprocessors = []
    for extractor in feature_extractors:
        logger.info("Computing relevance for extractors %s", extractor.name)
        extractor.extract_features()
        p = extractor.postprocessing(working_dir=results_dir,
                                     pdb_file=working_dir + "/trajectories/protein_noh.pdb",
                                     feature_to_resids=feature_to_resids,
                                     filter_results=False)
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


simu_type = "clustering"
for nclusters in range(6, 7):
    run(nclusters=nclusters,
        feature_type="ca_inv",
        simu_type=simu_type,
        n_iterations=1,
        n_splits=1,
        supervised=True,
        shuffle_datasets=False,
        filter_by_distance_cutoff=True)
    if simu_type != "clustering":
        break
