
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

logger = logging.getLogger("terminal")
working_dir = os.path.expanduser("~/kex/bachelor_thesis2019_gpcr/")     # Path to directory containing the data to be analysed
traj_dir = working_dir + "swarm_trajectories/" #TODO change directory
save_dir = working_dir + "analysis/data/"
logger.info("Done with init")

# Load the MD trajectories
stride = 50

apo_asp_traj = md.load(traj_dir + "asp79-apo-swarms-nowater-nolipid.xtc",
                        top=traj_dir + "asp79-apo-swarms-nowater-nolipid.pdb", stride=stride)

holo_asp_traj = md.load(traj_dir + "asp79-holo-swarms-nowater-nolipid.xtc",
                        top=traj_dir + "asp79-holo-swarms-nowater-nolipid.pdb", stride=stride)

logger.info("Loaded trajectories with properties %s, %s", holo_asp_traj, apo_asp_traj)


### Compute the interatomic distances for mdtrajectories ###
holo_asp_distances, holo_asp_residue_pairs = md.compute_contacts(holo_asp_traj,
                                   contacts="all",
                                   scheme="ca", #You may want to use 'ca'
                                   ignore_nonprotein=True)

apo_asp_distances, apo_asp_residue_pairs = md.compute_contacts(apo_asp_traj,
                                   contacts="all",
                                   scheme="ca", #You may want to use 'ca'
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

# samples.save()
np.save(save_dir+"sample.npy", samples)
np.save(save_dir+"labels.npy", labels)
np.save(save_dir+"feature_to_resids.npy", feature_to_resids)
logger.info("Samples generated and saved.")
