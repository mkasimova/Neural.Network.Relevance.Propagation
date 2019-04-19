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

    projection = []
    class_score = []

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
    fe.PCAFeatureExtractor(**kwargs),
    fe.RbmFeatureExtractor(n_components=100, **kwargs),
    fe.RandomForestFeatureExtractor(**kwargs),
    fe.KLFeatureExtractor(**kwargs),
    fe.MlpFeatureExtractor(hidden_layer_sizes=(100,), **kwargs)
    ]

    for extractor in feature_extractors:
        extractor.extract_features()
        if np.any(np.isnan(extractor.feature_importance)):
            projection.append([-5,-5])
            class_score.append([-5,-5,-5])
        else:
            pp = extractor.postprocessing(predefined_relevant_residues=moved_atoms, rescale_results=True, filter_results=False)
            pp.average().evaluate_performance()
            projection.append([pp.data_projector.separation_score, pp.data_projector.projection_class_entropy])
            if pp.auc is not None:
                print("AUC, POS_ratio and NEG_ratio: ", pp.auc, pp.pos_ratio, pp.neg_ratio)
                class_score.append([pp.auc, pp.pos_ratio, pp.neg_ratio])

    return projection, class_score

####

n_iter_per_example = 10
n_splits = 1
n_iterations = 5

#test_model = ['linear','non-linear','non-linear-random-displacement','non-linear-p-displacement']
test_model = ['non-linear-p-displacement']
test_noise = [1e-2,1e-2,2e-1,2e-1]

n_feature_extractors = 5
n_supervised_feature_extractors = 3
projection = np.zeros((n_feature_extractors,len(test_model),len(test_noise),n_iter_per_example,2))
class_score = np.zeros((n_supervised_feature_extractors,len(test_model),len(test_noise),n_iter_per_example,3))

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

            projection[:,i,j,k,:], class_score[:,i,j,k,:] = run_all_feature_extractors(data,cluster_indices,n_splits,n_iterations,dg.moved_atoms)

work_dir = '/media/mkasimova/Data2/NN_VSD/toy_model/testing.different.toy.models/Febr.12.updated.code/'

np.save(work_dir+'projection.CONTACTS.NON-LINEAR-P-DISPLACEMENT.npy',projection)
np.save(work_dir+'class_score.CONTACTS.NON-LINEAR-P-DISPLACEMENT.npy',class_score)

