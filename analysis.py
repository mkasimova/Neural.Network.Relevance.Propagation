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
import numpy as np
import mdtraj as md
from modules import utils, feature_extraction as fe, postprocessing, visualization
from modules.data_generation import DataGenerator
from modules import filtering, data_projection as dp
import matplotlib.pyplot as plt
import seaborn as sns
logger = logging.getLogger("terminal")

working_dir  = os.path.expanduser("~/kex/bachelor_thesis2019_gpcr/")  # Path to directory containing the data to be analysed
topology = md.load(working_dir + "/swarm_trajectories/asp79-apo-swarms-nowater-nolipid.pdb").topology # Load topology in order to map important features to residues
samples = np.load(working_dir + "analysis/data/sample.npy") # (Frames, features)
labels = np.load(working_dir + "analysis/data/labels.npy") # Labels: HOLO = 1, APO = 2
feature_to_resids = np.load(working_dir + "analysis/data/feature_to_resids.npy") # This array tells us which residues the index of a certain feature correspond to.
logger.info("Done loading files.")


# Method to get the name of the residue from the biological index resid
def get_residue(resid):
    res = topology.select("resSeq " + str(resid) + " and name CA")
    return topology.atom(res[0]).residue


def scatterplot():
    methods = ['MLP', 'RF', 'KL']
    sns.set()
    fig, axs = plt.subplots(ncols = len(methods))
    fig.suptitle('Contact pairs with highest importance')
    for i in range(len(methods)):
        feature_importance = np.load(working_dir + "analysis/" + methods[i] + "/feature_importance.npy") # feature_importance from training
        # print(feature_to_resids.shape, feature_importance[:,0].shape,"\n")

        # Find the feature_idx of the two most important features (contact pairs)
        feature_idx = feature_importance[:,0].argsort(axis=0)[[-1, -2]] # Get the feature_idx of the 2 most important features (inputs)

        print(feature_importance)

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
        axs[i].legend()
    plt.show()

scatterplot()



# # Feature importance vs frame plot
# plt.figure(2)
# plt.title('Most important feature')
# plt.plot( np.arange(np.argwhere(labels==2)[0][0]-1),  holo_feature1, 'b*', label='Holo')
# plt.plot( np.arange(samples.shape[0] - np.argwhere(labels==2)[0][0]), apo_feature1, 'ro',label='Apo')
# # plt.ylim(2, 2.5)
# plt.xlabel('Frame')
# plt.ylabel("Contacts for the pairs " + str(contacts1) + " (Å)")
# plt.legend()
# plt.show()
