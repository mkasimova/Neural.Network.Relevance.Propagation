from __future__ import absolute_import, division, print_function
import logging
import sys
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

from modules.data_generation import DataGenerator

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


