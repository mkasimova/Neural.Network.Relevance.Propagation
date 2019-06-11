from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import mdtraj as md
import numpy as np
import glob
from modules import utils, filtering, feature_extraction as fe, visualization, traj_preprocessing as tp

logger = logging.getLogger("beta2")
utils.remove_outliers = False


def _get_important_residues(supervised):
    npxxy = [322, 323, 324, 325]
    yy = [219, 326]
    connector = [121, 282]
    # g_protein = [131, 266, 268, 327, 272, 124, 279]
    # see https://www.nature.com/articles/srep34736/figures/3
    all_ligands = [109, 113, 114, 117, 193, 195, 203, 204, 207, 286, 289, 290, 293, 308, 309, 312]
    agonists = [193, 117, 109, 113, 308, 293, 289, 207]
    asp_cavity = [79]
    # For holo the ion is most prominently bound to Cys184, Asn187 and Cys190, as it should be.
    # There is occasionally a sodium bound to Asp192 and quite often a sodium bount to Asp300 but the signal is not near as strong.
    sodium_sites = [79, 113, 184, 187, 190, 192, 300, 319]
    identified_byt_not_known = [144, 160, 163, 169, 179, 316]
    other = [75, 82, 275]
    if supervised:
        return {
            # g_protein,
            #               npxxy,
            #               yy,
            # connector,
            'Ligand interactions': all_ligands,
            'Asp79': asp_cavity,
            'Glu268': [268],
            'Leu144': [144],
            # agonists,
            # identified_byt_not_known,
            # sodium_sites,
            # other
        }
    else:
        return {
            # 'G protein site': g_protein,
            'NPxxY': npxxy,
            'Glu268': [268],
            #'YY bond': yy,
            # 'Connector': connector,
            # 'Asp79': asp_cavity,
            # all_ligands,
            # agonists,
            # identified_byt_not_known,
            # sodium_sites,
            # other
        }
    return


def _load_trajectory_for_predictions(simu_type, ligand):
    if simu_type != "apo-holo" or ligand not in ['apo', 'holo']:
        raise NotImplementedError
    infile = "/home/oliverfl/MEGA/PHD/projects/relevance_propagation/results/apo-holo/trajectories/asp79-{}-swarms-nowater-nolipid".format(
        ligand)
    traj = md.load(infile + ".xtc", top=infile + ".pdb")
    samples, feature_to_resids, pairs = tp.to_distances(traj)
    return samples, None


def run(nclusters=2,
        simu_type="apo-holo",
        ligand_type='holo',
        n_iterations=1,
        n_splits=1,
        shuffle_datasets=True,
        overwrite=False,
        dt=10,
        feature_type="ca_inv",  # "contacts_5_cutoff", "closest-heavy_inv" or "CA_inv", "cartesian_ca", "cartesian_noh"
        filetype="svg",
        supervised=True,
        load_trajectory_for_predictions=True,
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
                               # variance_cutoff='auto',
                               variance_cutoff='1_components',
                               name='PCA',
                               **kwargs),
        # fe.RbmFeatureExtractor(classifier_kwargs={'n_components': 1},
        #                        relevance_method='from_lrp',
        #                        name='RBM',
        #                        **kwargs),
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
    if load_trajectory_for_predictions:
        other_samples, other_labels = _load_trajectory_for_predictions(simu_type, ligand_type)
    else:
        other_samples, other_labels = None, None
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
        fe.KLFeatureExtractor(**kwargs),
        # fe.MlpFeatureExtractor(
        #     name="MLP" if other_samples is None else "MLP_predictor_{}".format(ligand_type),
        #     classifier_kwargs={
        #         # 'hidden_layer_sizes': [int(min(100, data.shape[1]) / (i + 1)) + 1 for i in range(3)],
        #         'hidden_layer_sizes': (30,),
        #         'max_iter': 10000,
        #         'alpha': 0.01,
        #         'activation': "relu"
        #     },
        #     per_frame_importance_samples=other_samples,
        #     per_frame_importance_labels=other_labels,
        #     per_frame_importance_outfile="/home/oliverfl/projects/gpcr/mega/Result_Data/beta2-dror/apo-holo/trajectories"
        #                                  "/mlp_perframe_importance_{}/"
        #                                  "{}_mlp_perframeimportance_{}clusters_{}cutoff.txt"
        #         .format(ligand_type, feature_type, nclusters, "" if filter_by_distance_cutoff else "no"),
        #     **kwargs),
    ]

    if supervised is None:
        feature_extractors = unsupervised_feature_extractors + supervised_feature_extractors
    else:
        feature_extractors = supervised_feature_extractors if supervised else unsupervised_feature_extractors
    logger.info("Done. using %s feature extractors", len(feature_extractors))
    highlighted_residues = _get_important_residues(supervised)
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
        n_iterations=30,
        n_splits=4,
        supervised=False,
        shuffle_datasets=True,
        overwrite=False,
        load_trajectory_for_predictions=False,
        ligand_type='apo',
        filter_by_distance_cutoff=False)
    if simu_type != "clustering":
        break
