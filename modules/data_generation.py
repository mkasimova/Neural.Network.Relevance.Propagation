from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np

logger = logging.getLogger("dataGen")

class DataGenerator(object):
    
    def __init__(self, natoms, nclusters, nframes_per_cluster, noise_level=0.1):
        """
        Class which generates artificial atoms, puts them into artifical clusters and adds noise to them
        :param natoms: number of atoms
        :param nclusters: number of clusters
        :param nframes_per_cluster:
        :param noise_leve: strength of noise to be added
        """
        if natoms < nclusters:
            raise Exception("Cannot have more clusters than atoms")
        self.natoms = natoms
        self.nfeatures = int(self.natoms*(self.natoms-1)/2)
        self.nclusters = nclusters
        self.nframes_per_cluster = nframes_per_cluster
        self.noise_level = noise_level
        
    def generate_clusters(self):
        """
        Generates actual cluster centers
        """
        cluster_centers = np.zeros((self.nclusters, self.natoms,3))
        moved_atoms = np.zeros((self.nclusters,)) - 1
        start_conf = generate_conformation(self.natoms)
        for c in range(self.nclusters):
            while True:
                conf = np.copy(start_conf)
                #randomly move one atom 
                atom_to_move = np.random.randint(self.natoms)
                if atom_to_move not in moved_atoms:
                    moved_atoms[c] = atom_to_move
                    conf[atom_to_move,:] += 10
                    break
            cluster_centers[c] = conf 
        self.cluster_centers = cluster_centers
        self.moved_atoms = moved_atoms
        
    def to_features(self, conf):
        dists = np.empty((self.nfeatures,))
        idx = 0
        for n1, coords1 in enumerate(conf):
            for n2 in range(n1 + 1, self.natoms):
                coords2 = conf[n2]
                dists[idx] = np.linalg.norm(coords1-coords2)
                idx += 1
        return dists
            
    def feature_to_resids(self):
        ftr = np.empty((self.nfeatures, 2))
        idx = 0
        for n1 in range(self.natoms):
            for n2 in range(n1 + 1, self.natoms):
                ftr[idx, 0] = n1
                ftr[idx, 1] = n2
                idx += 1
        return ftr
        
    def generate_frames(self):
        """
        Generates and returns artificial frames with noise for the clusters
        """
        labels = np.zeros((self.nframes_per_cluster*self.nclusters, self.nclusters))
        data = np.zeros((self.nframes_per_cluster*self.nclusters, self.nfeatures))
        frame_idx = 0
        for f in range(self.nframes_per_cluster):
            for c in range(self.nclusters):
                labels[frame_idx, c] = 1
                conf = perturb(self.cluster_centers[c], self.noise_level)
                features = self.to_features(conf)
                data[frame_idx,:] = features
                frame_idx += 1
        return data, labels
        
def generate_conformation(natoms):
    conf = np.zeros((natoms, 3))
    for n in range(natoms):
        conf[n] = (np.random.rand(3,)*2 -1) #distributed between [-1,1]
    return conf


def perturb(conf, noise_level):
    natoms = len(conf)
    conf = np.copy(conf)
    for n in range(natoms):
        conf[n] += (np.random.rand(3,)*2 - 1)*noise_level  
    return conf
    