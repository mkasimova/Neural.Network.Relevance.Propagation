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
import mdtraj as md
from modules import utils, feature_extraction as fe, postprocessing, visualization
from modules.data_generation import DataGenerator
from modules import filtering, data_projection as dp
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger("terminal")
working_dir = os.path.expanduser("~/kex/bachelor_thesis2019_gpcr/")  # Path to directory containing the data to be classified
save_dir = working_dir + "analysis/data/"
traj_dir = working_dir + "swarm_trajectories/"
logger.info("Done with init")


### LOAD SAMPLES FROM FILE ###
samples = np.load(save_dir+"sample.npy")
labels = np.load(save_dir+"labels.npy")
feature_to_resids = np.load(save_dir+"feature_to_resids.npy")
logger.info("Done with loading samples from file.")

### FEATURE EXTRACTION ###
n_iterations, n_splits = 1, 1 #Number of times to run and number of splits in cross validation
filter_by_distance_cutoff = False #Remove all distances greater than 0.5 nm (configurable limit). Typically residues close to each other contribute most to the stability of the protein
use_inverse_distances = True #Usually it is a good idea to take the inverse of the distances since a larg number then indicates two residues in contact -> stronger interaction

feature_extractors = [
     fe.MlpFeatureExtractor(samples, labels, n_splits=n_splits, n_iterations=n_iterations,
                             hidden_layer_sizes=(100,100,100), # Number of neurons in each layer, e.g. (100,10) would be a bilayer perceptron with 100 neurons in the first layer and 10 in the second
                             activation="relu", #use "relu" or logistic(sigmoid), determines neuron output
                             solver="adam",
                             randomize=True, # set to false for reproducability
                             filter_by_distance_cutoff=filter_by_distance_cutoff),

     fe.RandomForestFeatureExtractor(samples, labels, n_splits=n_splits, n_iterations=n_iterations,
                             filter_by_distance_cutoff=filter_by_distance_cutoff),

     fe.KLFeatureExtractor(samples, labels, n_splits=n_splits,
                            filter_by_distance_cutoff=filter_by_distance_cutoff),

]

# ############ Unsupervised learning methods, skip them for a start#######################
#      fe.RbmFeatureExtractor(rbm_data, cluster_indices, n_splits=n_splits, n_iterations=n_iterations,
#                           n_components=8,
#                            use_inverse_distances=use_inverse_distances,
#                           filter_by_distance_cutoff=filter_by_distance_cutoff),
#      fe.PCAFeatureExtractor(data, cluster_indices, n_splits=n_splits,
#                            filter_by_distance_cutoff=filter_by_distance_cutoff) ]

logger.info("Done. using %s feature extractors", len(feature_extractors))


### RUN FEATURE EXTRACTION ###
results = []
for extractor in feature_extractors:
    logger.info("Computing relevance for extractors %s", extractor.name)
    feature_importance, std_feature_importance, errors = extractor.extract_features()
    #logger.info("Get feature_importance and std of shapes %s, %s", feature_importance.shape, std_feature_importance.shape)
    results.append((extractor, feature_importance, std_feature_importance, errors))
logger.info("Done")


### POSTPROCESSING ###
postprocessors = []
for (extractor, feature_importance, std_feature_importance, errors) in results:
    p = postprocessing.PostProcessor(extractor, feature_importance, std_feature_importance, errors, labels,
                                     working_dir  + "analysis/",
                                     pdb_file=traj_dir + "asp79-apo-swarms-nowater-nolipid.pdb",
                                     feature_to_resids=feature_to_resids,
                                     filter_results=True)
    p.average()
    p.evaluate_performance()
    p.persist()
    postprocessors.append([p])
logger.info("Done")


### VISUALIZE THE RESULTS ###
sns.set()
visualization.visualize(postprocessors,
          show_importance=True,
          show_performance=False,
          show_projected_data=False)

logger.info("Done. The settings were n_iterations, n_splits = %s, %s.\nFiltering (filter_by_distance_cutoff = %s)",
            n_iterations, n_splits, filter_by_distance_cutoff)