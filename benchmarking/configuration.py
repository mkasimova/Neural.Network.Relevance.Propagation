from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from modules import feature_extraction as fe


def get_feature_extractors_names(extractor_type, n_splits, n_iterations):
    return np.array([e.name for e in
                      create_feature_extractors(extractor_type, np.zeros((3, 4)), np.array([0, 1, 0]), n_splits,
                                                n_iterations)])


def create_feature_extractors(extractor_type, samples, cluster_indices, n_splits, n_iterations):
    extractor_kwargs = {
        'samples': samples,
        'cluster_indices': cluster_indices,
        'filter_by_distance_cutoff': False,
        'use_inverse_distances': True,
        'n_splits': n_splits,
        'n_iterations': n_iterations,
        'scaling': True
    }
    if extractor_type == "KL":
        return create_KL_feature_extractors(extractor_kwargs)
    elif extractor_type == "RF":
        return create_RF_feature_extractors(extractor_kwargs)
    elif extractor_type == "RBM":
        return create_RBM_feature_extractors(extractor_kwargs)
    elif extractor_type == "MLP":
        return create_MLP_feature_extractors(extractor_kwargs)
    elif extractor_type == "AE":
        return create_AE_feature_extractors(extractor_kwargs)
    elif extractor_type == "PCA":
        return create_PCA_feature_extractors(extractor_kwargs)
    else:
        raise Exception("Unknown extractor type {}".format(extractor_type))


def create_KL_feature_extractors(extractor_kwargs, bin_widths=[0.01, 0.1, 0.2, 0.5]):
    feature_extractors = [
        fe.KLFeatureExtractor(name="auto-width", **extractor_kwargs)
    ]
    for bin_width in bin_widths:
        ext = fe.KLFeatureExtractor(name="{}-width".format(bin_width), bin_width=bin_width, **extractor_kwargs)
        feature_extractors.append(ext)
    return feature_extractors


def create_RF_feature_extractors(extractor_kwargs, n_estimators=[10, 100, 200]):
    return [
        fe.RandomForestFeatureExtractor(
            name="{}-estimators".format(nest),
            classifier_kwargs={
                'n_estimators': nest
            },
            **extractor_kwargs)
        for nest in n_estimators
    ]


def create_PCA_feature_extractors(extractor_kwargs, variance_cutoffs=["auto", "1_components", "1_components", 50, 100]):
    return [
        fe.PCAFeatureExtractor(
            name="{}-cutoff".format(cutoff),
            variance_cutoff=cutoff,
            **extractor_kwargs)
        for cutoff in variance_cutoffs
    ]


def create_RBM_feature_extractors(extractor_kwargs, n_components=[1, 10, 100, 200]):
    return [
        fe.RbmFeatureExtractor(
            name="{}-components".format(ncomp),
            classifier_kwargs={
                'n_components': ncomp
            },
            **extractor_kwargs
        )
        for ncomp in n_components
    ]


def create_MLP_feature_extractors(extractor_kwargs,
                                  alpha_hidden_layers=[
                                      # Benchmarking layer size
                                      (0.0001, [10, ]),
                                      (0.0001, [1000, ]),
                                      (0.0001, [100, ]),  # actually used in both benchmarks
                                      (0.0001, [50, 10]),
                                      (0.0001, [30, 10, 5]),
                                      # benchmarking alpha
                                      (0.001, [100, ]),
                                      (0.01, [100, ]),
                                      (0.1, [100, ]),

                                  ]
                                  ):
    feature_extractors = []
    for alpha, layers in alpha_hidden_layers:
        name = "{}-alpha_{}-layers".format(alpha, "x".join([str(l) for l in layers]))
        feature_extractors.append(
            fe.MlpFeatureExtractor(
                name=name,
                classifier_kwargs={
                    'alpha': alpha,
                    'hidden_layer_sizes': layers,
                    'max_iter': 100000,
                    'solver': "adam"
                },
                activation="relu",
                **extractor_kwargs)
        )
    return feature_extractors


def create_AE_feature_extractors(extractor_kwargs,
                                 alpha_hidden_layers=[
                                     # Benchmarking layer size
                                     (0.0001, [10, 2, 10, ]),
                                     (0.0001, [100, 25, 100, ]),  # actually used in both benchmarks
                                     (0.0001, [100, 25, 5, 25, 100, ]),
                                     # benchmarking alpha
                                     (0.001, [100, 25, 100, ]),
                                     (0.01, [100, 25, 100, ]),
                                     (0.1, [100, 25, 100, ]),

                                 ]
                                 ):
    feature_extractors = []
    for alpha, layers in alpha_hidden_layers:
        name = "{}-alpha_{}-layers".format(alpha, "x".join([str(l) for l in layers]))
        feature_extractors.append(
            fe.MlpAeFeatureExtractor(
                name=name,
                classifier_kwargs={
                    'alpha': alpha,
                    'hidden_layer_sizes': layers,
                    'max_iter': 100000,
                    'solver': "adam"
                },
                activation="logistic",
                **extractor_kwargs)
        )
    return feature_extractors
