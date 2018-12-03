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
from modules import utils, feature_extraction as fe, postprocessing as pp
import modules.data_generation as data_generation
from modules.data_generation import DataGenerator

logger = logging.getLogger("dataGenNb")

####

def run_all_feature_extractors(data,cluster_indices):

    n_iterations, n_splits = 5, 1
    feature_extractors = [
    fe.MlpFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, hidden_layer_sizes=(100,), training_max_iter=10000, activation="relu"),
    fe.RbmFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations, n_components=8),
    fe.KLFeatureExtractor(data, cluster_indices, n_splits=n_splits),
    fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_components=None),
    fe.RandomForestFeatureExtractor(data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations),
    ]


    results = []
    for extractor in feature_extractors:
        extractor.error_limit = 5
        feature_importance, std_feature_importance, errors = extractor.extract_features()
        results.append((extractor, feature_importance, std_feature_importance, errors))


    postprocessors = []
    for (extractor, feature_importance, std_feature_importance, errors) in results:
        p = pp.PostProcessor(extractor, feature_importance, std_feature_importance, errors, cluster_indices,
                         working_dir='/media/mkasimova/Data2/NN_VSD/toy_model/testing.different.toy.models/many.points/', feature_to_resids=None,
                         filter_results=False,\
                         predefined_relevant_residues=dg.moved_atoms)
        p.average()
        postprocessors.append(p)

####

n_iter_per_example = 20

test_model = ['linear','non-linear','non-linear-random-displacement','non-linear-p-displacement']

for i_model in test_model:

    for nl in [1e-2,1e-1]:

        for i_iter in range(n_iter_per_example):

            dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[8,8,8], nframes_per_cluster=1200,\
                               noise_natoms=None, noise_level=nl,\
                               displacement=0.1,\
                               cluster_generation_method=test_model)
            dg.select_atoms_to_move()
            data, labels = dg.generate_frames()
            cluster_indices = labels.argmax(axis=1)

            run_all_feature_extractors(data,cluster_indices)

