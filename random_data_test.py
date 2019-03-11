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

dg = DataGenerator(natoms=40, nclusters=4, natoms_per_cluster=[1, 1, 1, 1], nframes_per_cluster=200,
                   noise_level=0.1,  # 1e-2, #1e-2,
                   displacement=0.1,
                   noise_natoms=None,
                   feature_type='compact-dist',  # carteesian_rot_trans
                   test_model='non-linear-random-displacement')
# dg.generate_frames()
# dg.generate_clusters()
# dg.select_atoms_to_move()
data, labels = dg.generate_data()
cluster_indices = labels.argmax(axis=1)
feature_to_resids = dg.feature_to_resids()
logger.info("Generated data of shape %s and %s clusters", data.shape, labels.shape[1])

kwargs = {
    'samples': data,
    'cluster_indices': cluster_indices,
    'filter_by_distance_cutoff': False,
    'use_inverse_distances': True,
    'n_splits': 1,
    'n_iterations': 1,
    # 'upper_bound_distance_cutoff': 1.,
    # 'lower_bound_distance_cutoff': 1.
}
variance_cutoff = "auto"
supervised_feature_extractors = [
    fe.MlpFeatureExtractor(hidden_layer_sizes=(dg.natoms, dg.nclusters*2),
        training_max_iter=10000,
        activation="logistic",
        **kwargs),
    fe.ElmFeatureExtractor(
        activation="relu",
        n_nodes=3 * dg.nfeatures,
        alpha=100,
        **kwargs),
    fe.KLFeatureExtractor(**kwargs),
    fe.RandomForestFeatureExtractor(one_vs_rest=True, **kwargs),
]
unsupervised_feature_extractors = [
    fe.MlpAeFeatureExtractor(
        hidden_layer_sizes=(dg.nclusters,),  # int(data.shape[1]/2),),
        # training_max_iter=10000,
        use_reconstruction_for_lrp=True,
        activation="logistic",
        **kwargs),

    fe.RbmFeatureExtractor(n_components=dg.nclusters,
                           relevance_method='from_components',
                           name='RBM_from_components',
                           variance_cutoff='auto',
                           **kwargs),
    fe.RbmFeatureExtractor(n_components=dg.nclusters,
                           relevance_method='from_lrp',
                           name='RBM',
                           **kwargs),
    fe.PCAFeatureExtractor(n_components=None,
                           variance_cutoff=101,
                           name='PCA_all',
                           **kwargs),
    fe.PCAFeatureExtractor(n_components=None,
                           name="PCA_%s" % variance_cutoff,
                           variance_cutoff=variance_cutoff,
                           **kwargs),
    fe.PCAFeatureExtractor(n_components=None,
                           variance_cutoff='6_components',
                           name='PCA_6_comp',
                           **kwargs),
]
feature_extractors = supervised_feature_extractors
logger.info("Done. using %s feature extractors", len(feature_extractors))

results = []
for extractor in feature_extractors:
    extractor.error_limit = 500
    logger.info("Computing relevance for extractors %s", extractor.name)
    extractor.extract_features()
    test_set_errors = extractor.test_set_errors
    feature_importance = extractor.feature_importance
    std_feature_importance = extractor.std_feature_importance

    # logger.info("Get feature_importance and std of shapes %s, %s", feature_importance.shape, std_feature_importance.shape)
    results.append((extractor, feature_importance, std_feature_importance, test_set_errors))
logger.info("Done")

postprocessors = []
filter_results = True
for (extractor, feature_importance, std_feature_importance, errors) in results:
    p = pp.PostProcessor(extractor,
                         working_dir=".",
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
                        show_projected_data=False,
                        outfile="output/test_importance_per_residue.svg")
visualization.visualize(postprocessors,
                        show_importance=False,
                        show_performance=True,
                        show_projected_data=False,
                        outfile="output/test_performance.svg")
visualization.visualize(postprocessors,
                        show_importance=False,
                        show_performance=False,
                        show_projected_data=True,
                        outfile="output/test_projection.svg")
logger.info("Done. The settings were n_iterations = {n_iterations}, n_splits = {n_splits}."
            "\nFiltering (filter_by_distance_cutoff={filter_by_distance_cutoff})".format(**kwargs))
