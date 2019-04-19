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
from modules import feature_extraction as fe, postprocessing as pp

def generate_data(i_model,j,j_noise,feature_type):
    if j%2==1:
        dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[10,10,10], nframes_per_cluster=1200, test_model=i_model,\
                           noise_natoms=12, noise_level=j_noise, displacement=0.1, feature_type=feature_type)
    else:
        dg = DataGenerator(natoms=100, nclusters=3, natoms_per_cluster=[10,10,10], nframes_per_cluster=1200, test_model=i_model,\
                           noise_natoms=None, noise_level=j_noise, displacement=0.1, feature_type=feature_type)
    data, labels = dg.generate_data()
    cluster_indices = labels.argmax(axis=1)
    moved_atoms = dg.moved_atoms
    feature_to_resids = dg.feature_to_resids()

    return data, cluster_indices, moved_atoms, feature_to_resids

####

logger = logging.getLogger("ToyModel")
work_dir = '/media/mkasimova/Data2/NN_VSD/toy_model/final.data.and.figures/'

test_model = ['linear','non-linear']
test_noise = [1e-2,1e-2,2e-1,2e-1]
n_iter_per_example = 20
#feature_type = 'inv-dist'
feature_type = 'cartesian_rot'

n_splits = 1
n_iterations = 5

# contacts
for i, i_model in enumerate(test_model):
    SCORES = None

    for j, j_noise in enumerate(test_noise):
        for k in range(n_iter_per_example):
            data, cluster_indices, moved_atoms, feature_to_resids = generate_data(i_model,j,j_noise,feature_type)

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
            fe.RbmFeatureExtractor(
                classifier_kwargs={
                'n_components': 100},
                **kwargs),
            fe.MlpAeFeatureExtractor(
                classifier_kwargs={
                'hidden_layer_sizes': (200, 100, 200,),
                'max_iter': 100000,
                'alpha': 0.0001,
                'solver': "adam"},
                **kwargs),
            fe.RandomForestFeatureExtractor(
                classifier_kwargs={
                'n_estimators': 1000},
                **kwargs),
            fe.KLFeatureExtractor(**kwargs),
            fe.MlpFeatureExtractor(
                classifier_kwargs={
                'hidden_layer_sizes': [100,],
                'max_iter': 100000,
                'alpha': 0.0001},
                activation="relu",
                **kwargs)
            ]

            if SCORES is None:
                SCORES = [[] for i_extractor in range(len(feature_extractors))]

            for i_extractor, extractor in enumerate(feature_extractors):
                extractor.extract_features()
                if np.any(np.isnan(extractor.feature_importance)):
                    if extractor.supervised:
                        SCORES[i_extractor].append([None,None,None,None,None])
                    else:
                        SCORES[i_extractor].append([None,None])
                else:
                    pp = extractor.postprocessing(predefined_relevant_residues=moved_atoms, rescale_results=True, filter_results=False, feature_to_resids=feature_to_resids)
                    pp.average().evaluate_performance()
                    if extractor.supervised:
                        SCORES[i_extractor].append([pp.auc_all_clusters,pp.accuracy_all_clusters,\
                                                    pp.auc_per_cluster,pp.accuracy_per_cluster,pp.data_projector.separation_score])
                    else:
                        SCORES[i_extractor].append([pp.auc_all_clusters,pp.accuracy_all_clusters])

    for i_extractor, extractor in enumerate(feature_extractors):
        np.save(work_dir+extractor.name+'.'+i_model+'.'+feature_type+'.auc_all_clusters.npy', np.asarray(SCORES[i_extractor])[:,0])
        np.save(work_dir+extractor.name+'.'+i_model+'.'+feature_type+'.accuracy_all_clusters.npy', np.asarray(SCORES[i_extractor])[:,1])
        if extractor.supervised:
            np.save(work_dir+extractor.name+'.'+i_model+'.'+feature_type+'.auc_per_cluster.npy', np.asarray(SCORES[i_extractor])[:,2])
            np.save(work_dir+extractor.name+'.'+i_model+'.'+feature_type+'.accuracy_per_cluster.npy', np.asarray(SCORES[i_extractor])[:,3])
            np.save(work_dir+extractor.name+'.'+i_model+'.'+feature_type+'.separation_score.npy', np.asarray(SCORES[i_extractor])[:,4])

