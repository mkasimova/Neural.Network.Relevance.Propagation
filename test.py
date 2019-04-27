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
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns


working_dir  = os.path.expanduser("~/kex/bachelor_thesis2019_gpcr/")  # Path to directory containing the data to be analysed
save_dir = working_dir + "analysis/"
traj_dir = working_dir + "swarm_trajectories/" #TODO change directory
stride = 50


apo_asp_traj = md.load(traj_dir + "ash79-holo-swarms-nowater-nolipid.xtc",
                        top=traj_dir + "ash79-holo-swarms-nowater-nolipid.pdb", stride=stride)

def get_residue(resid):
  ''' Method to get the name of the residue from the biological index resid '''
  res = apo_asp_traj.top.select("resSeq " + str(resid) + " and name CA")
  return apo_asp_traj.top.atom(res[0]).residue

def average_error(errors):
  ''' Method which returns an array of errors less than the error limit (5) and the average '''
  avg_error = [e for e in errors if e <= 5]
  return np.average(avg_error), 1-(len(errors)-len(avg_error))/len(errors)

def cluster_distance(X, Y):

    # Create (x,y) pairs
    if len(X) != len(Y):
        raise Exception('Given coordinates have different shapes, %s vs %s', len(X), len(Y))
    else:
        # Calculate average distance between points in cluster
        mean_distance = 0
        for i in range(len(X)):
            mean_distance += np.sqrt( (X[0]-X[i])**2 + (Y[0]-Y[i])**2 ) / len(X)
    return mean_distance


feature_to_resids = np.load(save_dir+"/data/feature_to_resids.npy")
samples = np.load(working_dir + "analysis/data/sample.npy") # (Frames, features)
labels = np.load(working_dir + "analysis/data/labels.npy") # Labels: HOLO = 1, APO = 2

# Load data files used for the processing
feature_importance = np.load(working_dir + "analysis/MLP/feature_importance.npy") # feature_importance from training
std_feature_importance = np.load(working_dir + "analysis/MLP/std_feature_importance.npy")



# Get the feature values from all frames for the top 1 and top 2 contacts
holo_feature1 = samples[labels==1, 0]  # X-coordinate
holo_feature2 = samples[labels==1, 1]  # Y-coordinate

apo_feature1 = samples[labels==2, 0]  # X-coordinate
apo_feature2 = samples[labels==2, 1]  # Y-coordinate

# Find the feature_idx of the two most important features (contact pairs)
feature_idx = feature_importance[:,0].argsort(axis=0)[[-1, -2, -3, -4, -5]] # Get the feature_idx of the 5 most important features (inputs)

resids1 = feature_to_resids[feature_idx[0]].astype(int)# Get the residue ids from top 1 contact pair
resids2 = feature_to_resids[feature_idx[1]].astype(int)# Get the residue ids from top 2 contact pair

# Get the residue names from the feature_idx
contacts1 = [get_residue(resids1[0]), get_residue(resids1[1])]
contacts2 = [get_residue(resids2[0]), get_residue(resids2[1])]

# Get the feature values from all frames for the top 1 and top 2 contacts
holo_feature1 = samples[labels==1, feature_idx[0]]  # Top 1 across all holo frames
holo_feature2 = samples[labels==1, feature_idx[1]]  # Top 2 across all holo frames

apo_feature1 = samples[labels==2, feature_idx[0]]  # Top 1 across all apo frames
apo_feature2 = samples[labels==2, feature_idx[1]]  # Top 2 across all apo frames


# Calculate the average distance between the points in each cluster
apo_cluster_avg_distance = cluster_distance(apo_feature1, apo_feature2)
holo_cluster_avg_distance = cluster_distance(holo_feature1, holo_feature2)

cluster_dist = "Mean cluster distance is " +str((apo_cluster_avg_distance+holo_cluster_avg_distance)/2)+"\nBlue: "+str(holo_cluster_avg_distance)+"\nRed: "+str(apo_cluster_avg_distance)


result = []
result.append(([1], [2], [3], [4]))
result.append((['a'], ['b'], ['c'], ['d']))
result.append(([1], [2], [3], [4]))

for i in range(3):
    print(result[i][3])

# Create an empty table with important features and the correspoding standard deviations
table = PrettyTable()
table.field_names = ['Feature Importance', 'Standard Deviation', 'Residue Pair']

# Get the data that is to be added to the table
top5_feats = feature_importance[:,0][feature_idx] # Get the relative importance of the top 5 features
std_feat = std_feature_importance[:,0][feature_idx] # Get the standard deviations associated with the top 5 features

# Build the ASCII table
for j in range(5):
  table.add_row([top5_feats[j], std_feat[j],  str([get_residue(feature_to_resids[feature_idx[j]].astype(int)[0]), get_residue(feature_to_resids[feature_idx[j]].astype(int)[1])]) ] )


# Write ASCII table, average distances and errors to ouput file


f = open(save_dir+"output.txt", "w") # Open the file and overwrite with new data
f.write(str(table)+"\n"+cluster_dist)
f.close()