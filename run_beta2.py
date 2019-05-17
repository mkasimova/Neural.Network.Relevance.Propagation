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
import glob
from modules import utils, filtering, feature_extraction as fe, visualization

logger = logging.getLogger("beta2")
utils.remove_outliers = False


def _get_important_residues():
    npxxy = [322, 323, 324, 325, 326, ]
    yy = [219, 326]
    connector = [121, 282]
    g_protein = [131, 266, 268, 327, 272, 124, 279]
    # see https://www.nature.com/articles/srep34736/figures/3
    all_ligands = [109, 113, 114, 117, 193, 195, 203, 204, 207, 286, 289, 290, 293, 308, 309, 312]
    agonists = [193, 117, 109, 113, 308, 293, 289, 207]
    asp_cavity = [79, 82, 322]
    other = [75, 275]
    all = [g_protein,
           yy,
           connector,
           npxxy,
           all_ligands,
           asp_cavity,
           agonists,
           other
           ]
    highlighted_residues = []
    for h in all:
        highlighted_residues += h
    highlighted_residues = set(highlighted_residues)
    return highlighted_residues


def run(nclusters=2,
        simu_type="apo-holo",
        n_iterations=1,
        n_splits=1,
        shuffle_datasets=True,
        overwrite=False,
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
        results_dir = "{}/results/{}/{}/".format(working_dir, feature_type,
                                                 "cutoff" if filter_by_distance_cutoff else "nocutoff")
        data = np.load("{}/samples/{}/samples_dt{}.npz".format(working_dir, feature_type, dt))['array']
        feature_to_resids = np.load("{}/samples/{}/feature_to_resids.npy".format(working_dir, feature_type))
        cluster_indices = np.loadtxt("{wd}/cluster_indices/cluster_indices_dt{dt}.txt".format(wd=working_dir, dt=dt))
    else:
        raise Exception("Unsupported simulation type {simu_type}".format(simu_type=simu_type))
    suffix = str(nclusters) + "clusters_" + str(n_iterations) + "iterations_" \
             + ("distance-cutoff_" if filter_by_distance_cutoff else "") + feature_type
    cluster_indices -= 1  # start at 0 instead of 1
    if len(data) != len(cluster_indices) or data.shape[1] != len(feature_to_resids):
        raise Exception("Inconsistent input data. The number of features or the number of frames to no match")
    logger.info("Loaded data of shape %s and %s clusters for %s clusters and feature type %s", data.shape,
                len(set(cluster_indices)), nclusters, feature_type)
    # ## Define the different methods to use
    #
    # Every method is encapsulated in a so called FeatureExtractor class which all follow the same interface
    cutoff_offset = 0.2 if "closest-heavy" in feature_type else 0
    kwargs = {
        'samples': data,
        'cluster_indices': cluster_indices,
        'filter_by_distance_cutoff': filter_by_distance_cutoff,
        'lower_bound_distance_cutoff': filtering.lower_bound_distance_cutoff_default - cutoff_offset,
        'upper_bound_distance_cutoff': filtering.upper_bound_distance_cutoff_default - cutoff_offset,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        'shuffle_datasets': shuffle_datasets
        # 'upper_bound_distance_cutoff': 1.,
        # 'lower_bound_distance_cutoff': 1.
    }
    unsupervised_feature_extractors = [
        fe.PCAFeatureExtractor(classifier_kwargs={'n_components': None},
                               variance_cutoff='auto',
                               name='PCA',
                               **kwargs),
        fe.RbmFeatureExtractor(classifier_kwargs={'n_components': 1},
                               relevance_method='from_lrp',
                               name='RBM',
                               **kwargs),
        # fe.MlpAeFeatureExtractor(
        #     classifier_kwargs={
        #         'hidden_layer_sizes': (100, 30, 2, 30, 100,),  # int(data.shape[1]/2),),
        #         # max_iter=10000,
        #         'alpha': 0.01,
        #         'activation': "logistic"
        #     },
        #     use_reconstruction_for_lrp=True,
        #     **kwargs),
    ]
    supervised_feature_extractors = [
        # fe.ElmFeatureExtractor(
        #     activation="relu",
        #     n_nodes=data.shape[1] * 2,
        #     alpha=0.1,
        #     **kwargs),
        fe.RandomForestFeatureExtractor(
            one_vs_rest=False,
            classifier_kwargs={'n_estimators': 1000},
            **kwargs),
        # fe.KLFeatureExtractor(**kwargs),
        # fe.MlpFeatureExtractor(
        #     classifier_kwargs={
        #         # 'hidden_layer_sizes': [int(min(100, data.shape[1]) / (i + 1)) + 1 for i in range(3)],
        #         'hidden_layer_sizes': (30,),
        #         'max_iter': 10000,
        #         'alpha': 0.01,
        #         'activation': "relu"
        #     },
        #     # per_frame_importance_outfile="/home/oliverfl/projects/gpcr/mega/Result_Data/beta2-dror/clustering_D09/trajectories"
        #     #                              "/mlp_perframe_importance/"
        #     #                              "{}_mlp_perframeimportance_{}clusters_{}cutoff.txt"
        #     #     .format(feature_type, nclusters, "" if filter_by_distance_cutoff else "no"),
        #     **kwargs),
    ]

    if supervised is None:
        feature_extractors = unsupervised_feature_extractors + supervised_feature_extractors
    else:
        feature_extractors = supervised_feature_extractors if supervised else unsupervised_feature_extractors
    logger.info("Done. using %s feature extractors", len(feature_extractors))
    highlighted_residues = _get_important_residues()
    # # Run the relevance analysis
    postprocessors = []
    for extractor in feature_extractors:
        do_computations = True
        if os.path.exists(results_dir):
            existing_files = glob.glob("{}/{}/importance_per_residue.npy".format(results_dir, extractor.name))
            if len(existing_files) > 0 and not overwrite:
                logger.debug("File %s already exists. skipping computations", existing_files[0])
                do_computations = False
        if do_computations:
            logger.info("Computing relevance for extractors %s", extractor.name)
            extractor.extract_features()
        p = extractor.postprocessing(working_dir=results_dir,
                                     pdb_file=working_dir + "/trajectories/all.pdb",
                                     # pdb_file=working_dir + "/trajectories/protein_noh.pdb",
                                     feature_to_resids=feature_to_resids,
                                     filter_results=False)
        if do_computations:
            p.average()
            p.evaluate_performance()
            p.persist()
        else:
            p.load()

        postprocessors.append([p])
        # # Visualize results
        visualization.visualize([[p]],
                                show_importance=True,
                                show_performance=False,
                                show_projected_data=False,
                                highlighted_residues=highlighted_residues,
                                outfile=results_dir + "/{extractor}/importance_per_residue_{suffix}_{extractor}.{filetype}".format(
                                    suffix=suffix,
                                    extractor=extractor.name,
                                    filetype=filetype))

        if do_computations:
            visualization.visualize([[p]],
                                    show_importance=False,
                                    show_performance=True,
                                    show_projected_data=False,
                                    outfile=results_dir + "/{extractor}/performance_{suffix}_{extractor}.{filetype}".format(
                                        suffix=suffix,
                                        extractor=extractor.name,
                                        filetype=filetype))
            visualization.visualize([[p]],
                                    show_importance=False,
                                    show_performance=False,
                                    show_projected_data=True,
                                    outfile=results_dir + "/{extractor}/projected_data_{suffix}_{extractor}.{filetype}".format(
                                        suffix=suffix,
                                        extractor=extractor.name,
                                        filetype=filetype))
    logger.info("Done. The settings were n_iterations = {n_iterations}, n_splits = {n_splits}."
                "\nFiltering (filter_by_distance_cutoff={filter_by_distance_cutoff})".format(**kwargs))


simu_type = "apo-holo"
for nclusters in range(2, 6):
    run(nclusters=nclusters,
        # feature_type="closest-heavy_inv",
        feature_type="ca_inv",
        simu_type=simu_type,
        n_iterations=10,
        n_splits=4,
        supervised=True,
        shuffle_datasets=True,
        overwrite=True,
        filter_by_distance_cutoff=False)
    if simu_type != "clustering":
        break
