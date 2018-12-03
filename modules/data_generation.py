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

#constant values
constant_displacement="constant_displacement"
random_displacement="random_displacement"
constant_and_random_displacement="contsant_and_random_displacement"

class DataGenerator(object):

    def __init__(self, natoms, nclusters, natoms_per_cluster, nframes_per_cluster, noise_natoms=None, noise_level=1e-3, cluster_generation_method=constant_displacement, displacement=10):
        """
        Class which generates artificial atoms, puts them into artifical clusters and adds noise to them
        :param natoms: number of atoms
        :param nclusters: number of clusters
        :param nframes_per_cluster:
        :param noise_level: strength of noise to be added
        :cluster_generation_method: 'constant_displacement' or 'random_displacement'
        :displacement: length of displacement vector for cluster generation 
        """
        if natoms < nclusters:
            raise Exception("Cannot have more clusters than atoms")
        self.natoms = natoms
        self.nfeatures = int(self.natoms*(self.natoms-1)/2)
        self.nclusters = nclusters
        self.natoms_per_cluster = natoms_per_cluster
        self.nframes_per_cluster = nframes_per_cluster
        self.noise_natoms = noise_natoms
        self.noise_level = noise_level
        self.cluster_generation_method = cluster_generation_method
        self.displacement = displacement

    def select_atoms_to_move(self):
        """
        Generates actual cluster centers
        """
        logger.info("Selecting atoms to move ...")
        moved_atoms_all_clusters = []
        for c in range(self.nclusters):
            moved_atoms_selected_cluster = []
            for a in range(self.natoms_per_cluster[c]):
                # Randomly select an atom
                while True:
                    atom_to_move = np.random.randint(self.natoms)
                    if (atom_to_move not in moved_atoms_selected_cluster) and (atom_to_move not in flatten(moved_atoms_all_clusters)):
                        moved_atoms_selected_cluster.append(atom_to_move)
                        break
            moved_atoms_all_clusters.append(moved_atoms_selected_cluster)
        self.moved_atoms = moved_atoms_all_clusters

        if self.noise_natoms is not None:
            moved_noise_atoms = []
            for a in range(self.noise_natoms):
                while True:
                    atom_to_move = np.random.randint(self.natoms)
                    if (atom_to_move not in moved_noise_atoms) and (atom_to_move not in flatten(moved_atoms_all_clusters)):
                        moved_noise_atoms.append(atom_to_move)
                        break
            self.moved_noise_atoms = moved_noise_atoms

    def to_features(self, conf):
        dists = np.empty((self.nfeatures,))
        idx = 0
        for n1, coords1 in enumerate(conf):
            for n2 in range(n1 + 1, self.natoms):
                coords2 = conf[n2]
                dists[idx] = np.linalg.norm(coords1-coords2) # Not inverse dist, is it ok?
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
        logger.info("Generating frames ...")
        start_conf = generate_conformation(self.natoms)
        labels = np.zeros((self.nframes_per_cluster*self.nclusters, self.nclusters))
        data = np.zeros((self.nframes_per_cluster*self.nclusters, self.nfeatures))
        frame_idx = 0
        for f in range(self.nframes_per_cluster):
            for c in range(self.nclusters):
                labels[frame_idx, c] = 1
                conf = np.copy(start_conf)

                ind_a = 0
                for a in self.moved_atoms[c]:
                    if self.cluster_generation_method == constant_displacement:
                        # Move one atom and let this be the cluster configuration
                        conf[a,:] += self.displacement
                    if self.cluster_generation_method == random_displacement:
                        # For every frame in a cluster we move the same atom, but we do it randomly with a random vector
                        theta, phi = np.random.rand()*2*np.pi, np.random.rand()*np.pi # Space angles
                        r  = self.displacement*np.random.rand()
                        vec = np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])
                        conf[a,:] += vec
                    if self.cluster_generation_method == constant_and_random_displacement:
                        decision = np.random.rand()
                        if decision>=0.5:
                            '''
                            if ind_a%4==0:
                                conf[a,:] += [self.displacement,self.displacement,self.displacement]
                            if ind_a%4==1:
                                conf[a,:] += [-self.displacement,self.displacement,self.displacement]
                            if ind_a%4==2:
                                conf[a,:] += [self.displacement,-self.displacement,self.displacement]
                            if ind_a%4==3:
                                conf[a,:] += [-self.displacement,-self.displacement,self.displacement]
                            '''
                            if ind_a%5==0:
                                conf[a,:] += [self.displacement,self.displacement,self.displacement]
                            if ind_a%5==1:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                conf[a,:] += [0,conf[a_minus_1,1]**2,0]
                            if ind_a%5==2:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                a_minus_2 = self.moved_atoms[c][ind_a-2]
                                conf[a,:] += [conf[a_minus_1,2]**2,0,conf[a_minus_2,0]**2]
                            if ind_a%5==3:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                conf[a,:] += [np.exp(conf[a_minus_1,1]),0.5*np.exp(conf[a_minus_1,1]),0]
                            if ind_a%5==4:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                a_minus_2 = self.moved_atoms[c][ind_a-2]
                                conf[a,:] += [conf[a_minus_1,0]*conf[a_minus_2,0]-np.sum(np.exp(conf[a,:])),0,self.displacement]
                    ind_a = ind_a+1

                if self.cluster_generation_method == constant_and_random_displacement and self.noise_natoms is not None:
                    for a in self.moved_noise_atoms:
                        #theta, phi = np.random.rand()*2*np.pi, np.random.rand()*np.pi
                        #theta, phi = 0, 0
                        #r  = 50*self.displacement*np.random.rand()
                        #vec = np.array([r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)])
                        #conf[a,:] += vec
                        if frame_idx%3==0:
                            conf[a,:] += 10*self.displacement

                conf = perturb(conf, self.noise_level)
                features = self.to_features(conf)
                data[frame_idx,:] = features
                frame_idx += 1
        return data, labels

def generate_conformation(natoms):
    conf = np.zeros((natoms, 3))
    for n in range(natoms):
        conf[n] = (np.random.rand(3,)*2 -1) # Distributed between [-1,1)
    return conf


def perturb(conf, noise_level):
    natoms = len(conf)
    for n in range(natoms):
        conf[n] += (np.random.rand(3,)*2 - 1)*noise_level  
    return conf

def flatten(list_of_lists):
    return [y for x in list_of_lists for y in x]
