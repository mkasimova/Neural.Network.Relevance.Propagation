from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
from modules import feature_extraction as fe, postprocessing as pp, visualization
from modules.data_generation import DataGenerator

logger = logging.getLogger("dataGenNb")

dg = DataGenerator(natoms=10, nclusters=3, natoms_per_cluster=[1, 1, 1], nframes_per_cluster=200,
                   noise_level=0.001,  # 1e-2, #1e-2,
                   displacement=0.5,
                   noise_natoms=None,
                   feature_type='dist',  # carteesian_rot_trans
                   test_model='linear')
# dg.generate_frames()
# dg.generate_clusters()
dg.select_atoms_to_move()
data, labels = dg.generate_frames()
cluster_indices = labels.argmax(axis=1)
feature_to_resids = dg.feature_to_resids()
logger.info("Generated data of shape %s and %s clusters", data.shape, labels.shape[1])

n_iterations, n_splits = 1, 1
variance_cutoff = "auto",
filter_by_distance_cutoff = False
feature_extractors = [
    # fe.MlpFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, #hidden_layer_sizes=(dg.natoms, dg.nclusters*2),
    #                       training_max_iter=10000,
    #                       activation="logistic",
    #                      filter_by_distance_cutoff=filter_by_distance_cutoff), #, solver="sgd"),
    # fe.MlpAeFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
    #                      hidden_layer_sizes=(dg.nclusters,),#int(data.shape[1]/2),),
    #                      #training_max_iter=10000,
    #                     use_reconstruction_for_lrp=False,
    #                     activation="logistic"), #, solver="sgd"),
    fe.RbmFeatureExtractor(data, cluster_indices,
                           n_splits=n_splits,
                           n_iterations=n_iterations,
                           n_components=dg.nclusters,
                           relevance_method='from_components',
                           variance_cutoff=variance_cutoff,
                           filter_by_distance_cutoff=filter_by_distance_cutoff),

    fe.RbmFeatureExtractor(data, cluster_indices,
                           n_splits=n_splits,
                           n_iterations=n_iterations,
                           n_components=dg.nclusters,
                           relevance_method='from_lrp',
                           variance_cutoff=variance_cutoff,
                           filter_by_distance_cutoff=filter_by_distance_cutoff),
    #     fe.ElmFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
    #                            activation="logistic",
    #                            n_nodes=3*dg.nfeatures,
    #                            alpha=1,
    #                            filter_by_distance_cutoff=filter_by_distance_cutoff),
    #      fe.KLFeatureExtractor(data, cluster_indices, n_splits=n_splits,
    #                             filter_by_distance_cutoff=filter_by_distance_cutoff),
    fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_components=None,
                           variance_cutoff=variance_cutoff,
                           filter_by_distance_cutoff=filter_by_distance_cutoff),
    fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_components=None,
                           variance_cutoff=1,
                           filter_by_distance_cutoff=filter_by_distance_cutoff),
    #     fe.RandomForestFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
    #                                                                filter_by_distance_cutoff=filter_by_distance_cutoff, one_vs_rest=True),
]
logger.info("Done. using %s feature extractors", len(feature_extractors))

results = []
for extractor in feature_extractors:
    extractor.error_limit = 500
    logger.info("Computing relevance for extractors %s", extractor.name)
    feature_importance, std_feature_importance, errors = extractor.extract_features()
    # logger.info("Get feature_importance and std of shapes %s, %s", feature_importance.shape, std_feature_importance.shape)
    results.append((extractor, feature_importance, std_feature_importance, errors))
logger.info("Done")

postprocessors = []
filter_results = True
for (extractor, feature_importance, std_feature_importance, errors) in results:
    p = pp.PostProcessor(extractor, feature_importance, std_feature_importance, errors, cluster_indices,
                         ".",
                         pdb_file=None,
                         feature_to_resids=feature_to_resids,
                         filter_results=filter_results)
    p.average()
    p.evaluate_performance()
    # p.persist()
    postprocessors.append([p])
logger.info("Done")

logger.info(
    "Actual atoms moved: %s.\n(Cluster generation method %s. Noise level=%s, displacement=%s. frames/cluster=%s)",
    sorted(dg.moved_atoms),
    dg.test_model, dg.noise_level, dg.displacement, dg.nframes_per_cluster)
visualization.visualize(postprocessors,
                        show_importance=True,
                        show_performance=False,
                        show_projected_data=False)
logger.info("Done. The settings were n_iterations, n_splits = %s, %s.\nFiltering (filter_by_distance_cutoff = %s)",
            n_iterations, n_splits, filter_by_distance_cutoff)
