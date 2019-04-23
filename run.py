#coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import os
import sys
import numpy as np
import mdtraj as md
from modules import utils, feature_extraction as fe, postprocessing, visualization
from modules.data_generation import DataGenerator
from modules import filtering, data_projection as dp
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

# Method to get the name of the residue from the biological index resid
def get_residue(resid):
    res = apo_asp_traj.top.select("resSeq " + str(resid) + " and name CA")
    return apo_asp_traj.top.atom(res[0]).residue

def average_error(errors):
  ''' Method which returns an array of errors less than the error limit (5) and the average '''
  avg_error = [e for e in errors if e <= 5]
  return np.average(avg_error), 1-(len(errors)-len(avg_error))/len(errors)


logger = logging.getLogger("terminal")
working_dir = os.path.expanduser("~/kex/bachelor_thesis2019_gpcr/")     # Path to directory containing the data to be analysed
traj_dir = working_dir + "swarm_trajectories/" #TODO change directory
save_dir = working_dir + "analysis/data/"
logger.info("Done with init")

# Load the MD trajectories
stride = 50


# ASP79 APO / HOLO
apo_asp_traj = md.load(traj_dir + "asp79-apo-swarms-nowater-nolipid.xtc",
                        top=traj_dir + "asp79-apo-swarms-nowater-nolipid.pdb", stride=stride)

holo_asp_traj = md.load(traj_dir + "asp79-holo-swarms-nowater-nolipid.xtc",
                        top=traj_dir + "asp79-holo-swarms-nowater-nolipid.pdb", stride=stride)

# ## ASH79 APO / HOLO
# apo_asp_traj = md.load(traj_dir + "ash79-apo-swarms-nowater-nolipid.xtc",
#                         top=traj_dir + "ash79-apo-swarms-nowater-nolipid.pdb", stride=stride)

# holo_asp_traj = md.load(traj_dir + "ash79-holo-swarms-nowater-nolipid.xtc",
#                         top=traj_dir + "ash79-holo-swarms-nowater-nolipid.pdb", stride=stride)

# # ASP79 APO  Neutral / Deprotonated
# apo_asp_traj = md.load(traj_dir + "asp79_Na-apo-swarms-nowater-nolipid.xtc",
#                         top=traj_dir + "asp79_Na-apo-swarms-nowater-nolipid.pdb", stride=stride)

# holo_asp_traj = md.load(traj_dir + "asp79-apo-swarms-nowater-nolipid.xtc",
#                         top=traj_dir + "asp79-apo-swarms-nowater-nolipid.pdb", stride=stride)


# ## ASP79 HOLO Neutral / Deprotonated
# apo_asp_traj = md.load(traj_dir + "asp79_Na-holo-swarms-nowater-nolipid.xtc",
#                         top=traj_dir + "asp79_Na-holo-swarms-nowater-nolipid.pdb", stride=stride)

# holo_asp_traj = md.load(traj_dir + "asp79-holo-swarms-nowater-nolipid.xtc",
#                         top=traj_dir + "asp79-holo-swarms-nowater-nolipid.pdb", stride=stride)

logger.info("Loaded trajectories with properties %s, %s", holo_asp_traj, apo_asp_traj)


### Compute the interatomic distances for mdtrajectories ###
holo_asp_distances, holo_asp_residue_pairs = md.compute_contacts(holo_asp_traj,
                                   contacts="all",
                                   scheme="closest-heavy", #You may want to use 'ca'
                                   ignore_nonprotein=True)

apo_asp_distances, apo_asp_residue_pairs = md.compute_contacts(apo_asp_traj,
                                   contacts="all",
                                   scheme="closest-heavy", #You may want to use 'ca'
                                   ignore_nonprotein=True)

if holo_asp_distances.shape[1] != apo_asp_distances.shape[1]:
    raise Exception("Different number of contacts in the two simulations. Must be the same")
logger.info("Done computing %s and %s distances", holo_asp_distances.shape, apo_asp_distances.shape)


#Merge this together into a data format suitable for training - samples and labels
samples = np.empty((len(holo_asp_distances) + len(apo_asp_distances), apo_asp_distances.shape[1]))
labels = np.empty((samples.shape[0],))
samples[0:len(holo_asp_distances)] = holo_asp_distances
samples[len(holo_asp_distances):] = apo_asp_distances
labels[0:len(holo_asp_distances)] = 1 # We label holo with '1'
labels[len(holo_asp_distances):] = 2 # and apo with '2'

# cluster_indices = []

# Map the indices of the samples to the right residue number

# Feature is defined as the contact between two residues
feature_to_resids = np.empty((samples.shape[1], 2)) #This array tells us which residues the index of a certain feature correspond to.
for feature_idx, (res1h, res2h) in enumerate(holo_asp_residue_pairs):
    res1a, res2a = apo_asp_residue_pairs[feature_idx]
    #Convert to biological residue number - this is independent of what your topology (which is a datastructure) looks like
    res1h = holo_asp_traj.top.residue(res1h).resSeq
    res2h = holo_asp_traj.top.residue(res2h).resSeq
    res1a = apo_asp_traj.top.residue(res1a).resSeq
    res2a = apo_asp_traj.top.residue(res2a).resSeq
    if res1h != res1a or res2h != res2a:
        raise Exception("Features differ at index %s. Must be aligned. (%s!=%s or %s!=%s)", feature_idx, res1h, res1a, res2h, res2a)
    else:
        feature_to_resids[feature_idx, 0] = res1h
        feature_to_resids[feature_idx, 1] = res2h

logger.info("Done. Created samples of shape %s", samples.shape)



### FEATURE EXTRACTION ###
n_iterations, n_splits = 25, 4 #Number of times to run and number of splits in cross validation
filter_by_distance_cutoff = False #Remove all distances greater than 0.5 nm (configurable limit). Typically residues close to each other contribute most to the stability of the protein
use_inverse_distances = True #Usually it is a good idea to take the inverse of the distances since a larg number then indicates two residues in contact -> stronger interaction

feature_extractors = [
     fe.MlpFeatureExtractor(samples, labels, n_splits=n_splits, n_iterations=n_iterations,
                             hidden_layer_sizes=(100,), # Number of neurons in each layer, e.g. (100,10) would be a bilayer perceptron with 100 neurons in the first layer and 10 in the second
                             activation="relu", #use "relu" or logistic(sigmoid), determines neuron output
                             solver="adam",
                             randomize=True, # set to false for reproducability
                             filter_by_distance_cutoff=filter_by_distance_cutoff),

     fe.RandomForestFeatureExtractor(samples, labels, n_splits=n_splits, n_iterations=n_iterations,
                             filter_by_distance_cutoff=filter_by_distance_cutoff),

     fe.KLFeatureExtractor(samples, labels, n_splits=n_splits,
                            filter_by_distance_cutoff=filter_by_distance_cutoff),


        ############ Unsupervised learning methods #######################
     # fe.RbmFeatureExtractor(samples, labels, n_splits=n_splits, n_iterations=n_iterations,
     #                      n_components=8,
     #                       # use_inverse_distances=use_inverse_distances,
     #                      filter_by_distance_cutoff=filter_by_distance_cutoff),
     # fe.PCAFeatureExtractor(samples, labels, n_splits=n_splits,
     #                       filter_by_distance_cutoff=filter_by_distance_cutoff)
]


logger.info("Done. using %s feature extractors", len(feature_extractors))


### RUN FEATURE EXTRACTION ###
results = []
for extractor in feature_extractors:
    logger.info("Computing relevance for extractors %s", extractor.name)
    feature_importance, std_feature_importance, errors = extractor.extract_features()
    #logger.info("Get feature_importance and std of shapes %s, %s", feature_importance.shape, std_feature_importance.shape)
    print("Average error and success rate for the method models: ", average_error(errors))
    results.append((extractor, feature_importance, std_feature_importance, errors))
logger.info("Done")


### POSTPROCESSING ###
postprocessors = []
for (extractor, feature_importance, std_feature_importance, errors) in results:
    p = postprocessing.PostProcessor(extractor, feature_importance, std_feature_importance, errors, labels,
                                     working_dir  + "analysis/",
                                     pdb_file=traj_dir + "asp79_Na-apo-swarms-nowater-nolipid.pdb",
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


methods = [extractor.name for extractor in feature_extractors]

# Print top 5 features of the methods, with std and contact pair

for j in range(len(methods)):

    table = PrettyTable()
    table.field_names = ['Feature Importance', 'Standard Deviation', 'Contact Pair']

    feature_importance = np.load(working_dir + "analysis/" + methods[j] + "/feature_importance.npy") # feature_importance from training
    std_feature_importance = np.load(working_dir + "analysis/" + methods[j] + "/std_feature_importance.npy")
    # print(feature_to_resids.shape, feature_importance[:,0].shape,"\n")

    # Find the feature_idx of the two most important features (contact pairs)
    feature_idx = feature_importance[:,0].argsort(axis=0)[[-1, -2, -3, -4, -5]] # Get the feature_idx of the 2 most important features (inputs)
    std_feat = std_feature_importance[:,0][feature_idx]
    top5_feats = feature_importance[:,0][feature_idx]

    for i in range(5):
      table.add_row([top5_feats[i], std_feat[i],  str([get_residue(feature_to_resids[feature_idx[i]].astype(int)[0]), get_residue(feature_to_resids[feature_idx[i]].astype(int)[1])]) ] )

    print(methods[j], '\n' ,table)

# Create scatterplots of top 2 features for each method

fig, axs = plt.subplots(ncols = len(methods))
fig.suptitle('Contact pairs with highest importance')
for i in range(len(methods)):
    feature_importance = np.load(working_dir + "analysis/" + methods[i] + "/feature_importance.npy") # feature_importance from training
    # print(feature_to_resids.shape, feature_importance[:,0].shape,"\n")

    # Find the feature_idx of the two most important features (contact pairs)
    feature_idx = feature_importance[:,0].argsort(axis=0)[[-1, -2]] # Get the feature_idx of the 2 most important features (inputs)

    resids1 = feature_to_resids[feature_idx[0]].astype(int)# Get the residue ids from top 1 contact pair
    resids2 = feature_to_resids[feature_idx[1]].astype(int)# Get the residue ids from top 2 contact pair

    # Get the residue names from the feature_idx
    contacts1 = [get_residue(resids1[0]), get_residue(resids1[1])]
    contacts2 = [get_residue(resids2[0]), get_residue(resids2[1])]

    # Get the feature values from all frames for the top 1 and top 2 contacts
    holo_feature1 = samples[labels==1, feature_idx[0]]  # Top 1 across all holo frames
    holo_feature2 = samples[labels==1, feature_idx[1]]  # Top 2 across all holo frames

    apo_feature1 = samples[labels==2, feature_idx [0]]  # Top 1 across all apo frames
    apo_feature2 = samples[labels==2, feature_idx [1]]  # Top 2 across all apo frames

    axs[i].set_title(methods[i])
    axs[i].scatter(holo_feature1, holo_feature2, c='blue', label='Holo')
    axs[i].scatter(apo_feature1, apo_feature2, c='red', label='Apo')
    axs[i].set_xlabel(u"Contacts for the pairs " + str(contacts1) + u" (Å)")
    axs[i].set_ylabel(u"Contacts for the pairs " + str(contacts2) + u" (Å)")
    axs[i].legend(loc=2)
plt.show()




