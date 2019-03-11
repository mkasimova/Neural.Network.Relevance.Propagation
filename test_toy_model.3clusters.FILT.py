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

#import modules.data_generation as data_generation
from modules.data_generation import DataGenerator
from modules import feature_extraction as fe, postprocessing as pp

logger = logging.getLogger("dataGenNb")

####

def run_all_feature_extractors(data,cluster_indices,n_splits,n_iterations,moved_atoms):

    projection_entropy = []
    area_under_roc = []

    feature_extractors = [
    fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_components=None, scaling=True, filter_by_distance_cutoff=False),
    fe.RbmFeatureExtractor(data, cluster_indices, n_components=100, n_splits=n_splits, n_iterations=n_iterations, scaling=True, filter_by_distance_cutoff=False),
    fe.RandomForestFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, scaling=True, filter_by_distance_cutoff=False),
    fe.KLFeatureExtractor(data, cluster_indices, n_splits=n_splits, scaling=True, filter_by_distance_cutoff=False),
    fe.MlpFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, hidden_layer_sizes=(100,), scaling=True, filter_by_distance_cutoff=False)
    ]

    for extractor in feature_extractors:
        extractor.extract_features()
        pp = extractor.postprocessing(predefined_relevant_residues=moved_atoms, rescale_results=True, filter_results=True)
        pp.average().evaluate_performance()
        projection_entropy.append(pp.data_projector.projection_class_entropy)
        if pp.auc is not None:
            area_under_roc.append(pp.auc)

    return projection_entropy, area_under_roc

####

n_iter_per_example = 10
n_splits = 1
n_iterations = 5

test_model = ['linear','non-linear','non-linear-random-displacement','non-linear-p-displacement']
test_noise = [1e-2,1e-2,2e-1,2e-1]

n_feature_extractors = 5
n_supervised_feature_extractors = 3
projection_entropy = np.zeros((n_feature_extractors,len(test_model),len(test_noise),n_iter_per_example))
area_under_roc = np.zeros((n_supervised_feature_extractors,len(test_model),len(test_noise),n_iter_per_example))

for i, i_model in enumerate(test_model):

    for j, j_noise in enumerate(test_noise):

        for k in range(n_iter_per_example):

            logger.info("i_model %s, j_noise %s", i_model, j_noise)

            # for j%2==1 use constant noise with noise_natoms=12
            if j%2==1:

                dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[10,10,10], nframes_per_cluster=1200, test_model=i_model,\
                               noise_natoms=12, noise_level=j_noise,\
                               displacement=0.1)

            else:

                dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[10,10,10], nframes_per_cluster=1200, test_model=i_model,\
                               noise_natoms=None, noise_level=j_noise,\
                               displacement=0.1)

            data, labels = dg.generate_data()
            cluster_indices = labels.argmax(axis=1)

            projection_entropy[:,i,j,k], area_under_roc[:,i,j,k] = run_all_feature_extractors(data,cluster_indices,n_splits,n_iterations,dg.moved_atoms)

work_dir = '/media/mkasimova/Data2/NN_VSD/toy_model/testing.different.toy.models/Febr.12.updated.code/'

np.save(work_dir+'projection_entropy.FILT.npy',projection_entropy)
np.save(work_dir+'area_under_roc.FILT.npy',area_under_roc)

