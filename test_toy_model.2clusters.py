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

from modules.data_generation import DataGenerator
from modules import utils, feature_extraction as fe, postprocessing as pp
from modules import filtering, data_projection as dp
from modules import comparison_bw_fe as comp_fe

logger = logging.getLogger("dataGenNb")

####

def run_all_feature_extractors(data,cluster_indices,n_splits,n_iterations,moved_atoms):

    feature_extractors = [
    fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_components=None),
    fe.RbmFeatureExtractor(data, cluster_indices, 10, n_splits=n_splits, n_iterations=n_iterations),
    fe.RandomForestFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations),
    fe.KLFeatureExtractor(data, cluster_indices, n_splits=n_splits),
    fe.ElmFeatureExtractor(data, cluster_indices, 1000, n_splits=n_splits, n_iterations=n_iterations),
    fe.MlpFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, hidden_layer_sizes=(100,)),
    ]


    results = []
    for extractor in feature_extractors:
        extractor.error_limit = 5
        feature_importance, std_feature_importance, errors = extractor.extract_features()
        results.append((extractor, feature_importance, std_feature_importance, errors))


    average_std = []
    projection_entropy = []
    area_under_roc = []
    for (extractor, feature_importance, std_feature_importance, errors) in results:
        p = pp.PostProcessor(extractor, feature_importance, std_feature_importance, errors, cluster_indices,
                         working_dir='/media/mkasimova/Data2/NN_VSD/toy_model/debug.after.cleaning/',
                         predefined_relevant_residues=moved_atoms)
        p.average().evaluate_performance()
        average_std.append(p.average_std)
        projection_entropy.append(p.data_projector.projection_class_entropy)
        area_under_roc.append(p.auc)

    return average_std, projection_entropy, area_under_roc

####

n_iter_per_example = 1
n_splits = 1
n_iterations = 1

test_model = ['linear','non-linear','non-linear-random-displacement','non-linear-p-displacement']
test_noise = [1e-2,1e-2,2e-1,2e-1]

n_feature_extractors = 6
average_std = np.zeros((n_feature_extractors,len(test_model),len(test_noise),n_iter_per_example))
projection_entropy = np.zeros((n_feature_extractors,len(test_model),len(test_noise),n_iter_per_example))
area_under_roc = np.zeros((n_feature_extractors,len(test_model),len(test_noise),n_iter_per_example))

for i, i_model in enumerate(test_model):

    for j, j_noise in enumerate(test_noise):

        for k in range(n_iter_per_example):

            logger.info("i_model %s, j_noise %s", i_model, j_noise)

            # for j%2==1 use constant noise with noise_natoms=12
            if j%2==1:

                dg = DataGenerator(natoms=100, nclusters=2, natoms_per_cluster=[12,12], nframes_per_cluster=1200, test_model=i_model,\
                               noise_natoms=12, noise_level=j_noise,\
                               displacement=0.1)

            else:

                dg = DataGenerator(natoms=100, nclusters=2, natoms_per_cluster=[12,12], nframes_per_cluster=1200, test_model=i_model,\
                               noise_natoms=None, noise_level=j_noise,\
                               displacement=0.1)

            dg.select_atoms_to_move()
            data, labels = dg.generate_frames()
            cluster_indices = labels.argmax(axis=1)

            average_std[:,i,j,k], projection_entropy[:,i,j,k], area_under_roc[:,i,j,k] = run_all_feature_extractors(data,cluster_indices,n_splits,n_iterations,dg.moved_atoms)


np.save('/media/mkasimova/Data2/NN_VSD/toy_model/debug.after.cleaning/average_std.all_tests_2clusters.PER_CLUSTER.npy',average_std)
np.save('/media/mkasimova/Data2/NN_VSD/toy_model/debug.after.cleaning/projection_entropy.all_tests_2clusters.PER_CLUSTER.npy',projection_entropy)
np.save('/media/mkasimova/Data2/NN_VSD/toy_model/debug.after.cleaning/area_under_roc.all_tests_2clusters.PER_CLUSTER.npy',area_under_roc)
