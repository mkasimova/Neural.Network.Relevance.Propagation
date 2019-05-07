from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

from modules.data_generation import DataGenerator
from modules import feature_extraction as fe


def generate_data(i_model, j, j_noise, feature_type):
    if j % 2 == 1:
        dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[10, 10, 10], nframes_per_cluster=1200,
                           test_model=i_model, \
                           noise_natoms=12, noise_level=j_noise, displacement=0.1, feature_type=feature_type)
    else:
        dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[10, 10, 10], nframes_per_cluster=1200,
                           test_model=i_model, \
                           noise_natoms=None, noise_level=j_noise, displacement=0.1, feature_type=feature_type)
    data, labels = dg.generate_data()
    cluster_indices = labels.argmax(axis=1)
    moved_atoms = dg.moved_atoms
    feature_to_resids = dg.feature_to_resids()

    return data, cluster_indices, moved_atoms, feature_to_resids


def create_feature_exractors(extractor_type, **kwargs):
    if extractor_type == "KL":
        return create_KL_feature_exractors(**kwargs)
    else:
        raise Exception("Unknown extractory type {}".format(extractor_type))


def create_KL_feature_exractors(data, cluster_indices, n_splits, n_iterations):
    kwargs = {
        'samples': data,
        'cluster_indices': cluster_indices,
        'filter_by_distance_cutoff': False,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        'scaling': True
    }

    feature_extractors = [
        fe.KLFeatureExtractor(name="auto", **kwargs)
    ]
    for bin_width in [0.01, 0.1, 0.2, 0.5]:
        ext = fe.KLFeatureExtractor(name="{}-width".format(bin_width), bin_width=bin_width, **kwargs)
        feature_extractors.append(ext)
    return feature_extractors
