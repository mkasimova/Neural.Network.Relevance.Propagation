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

    def __init__(self, natoms, nclusters, natoms_per_cluster, nframes_per_cluster, test_model='linear', noise_level=1e-2, noise_natoms=None, displacement=0.1):
        """
        Class which generates artificial atoms, puts them into artifical clusters and adds noise to them
        :param natoms: number of atoms
        :param nclusters: number of clusters
        :param nframes_per_cluster:
        :param test_model: 'linear','non-linear','non-linear-random-displacement','non-linear-p-displacement'
        :param noise_level: strength of noise to be added
        :param noise_natoms: number of atoms for constant noise
        :param displacement: length of displacement vector for cluster generation 
        """
        if natoms < nclusters:
            raise Exception("Cannot have more clusters than atoms")
        self.natoms = natoms
        self.nfeatures = int(self.natoms*(self.natoms-1)/2)
        self.nclusters = nclusters
        self.natoms_per_cluster = natoms_per_cluster
        self.nframes_per_cluster = nframes_per_cluster
        self.test_model = test_model
        self.noise_level = noise_level
        self.noise_natoms = noise_natoms
        self.displacement = displacement

    def select_atoms_to_move(self):

        """
        Generates actual cluster centers
        """
        logger.info("Selecting atoms to be moved ...")
        moved_atoms_all_clusters = []
        for c in range(self.nclusters):
            moved_atoms_selected_cluster = []
            for a in range(self.natoms_per_cluster[c]):
                # Randomly select an atom
                while True:
                    atom_to_move = np.random.randint(self.natoms)
                    if (atom_to_move not in moved_atoms_selected_cluster) and (atom_to_move not in self._flatten(moved_atoms_all_clusters)):
                        moved_atoms_selected_cluster.append(atom_to_move)
                        break
            moved_atoms_all_clusters.append(moved_atoms_selected_cluster)
        self.moved_atoms = moved_atoms_all_clusters

        if self.noise_natoms is not None:
            moved_noise_atoms = []
            for a in range(self.noise_natoms):
                while True:
                    atom_to_move = np.random.randint(self.natoms)
                    if (atom_to_move not in moved_noise_atoms) and (atom_to_move not in self._flatten(moved_atoms_all_clusters)):
                        moved_noise_atoms.append(atom_to_move)
                        break
            self.moved_noise_atoms = moved_noise_atoms

    def _flatten(self,list_of_lists):
        return [y for x in list_of_lists for y in x]

    def generate_frames(self):
        """
        Generates and returns artificial frames with noise for the clusters
        """
        logger.info("Generating frames ...")
        start_conf = self._generate_conformation()
        labels = np.zeros((self.nframes_per_cluster*self.nclusters, self.nclusters))
        data = np.zeros((self.nframes_per_cluster*self.nclusters, self.nfeatures))
        frame_idx = 0
        for f in range(self.nframes_per_cluster):
            for c in range(self.nclusters):
                labels[frame_idx, c] = 1
                conf = np.copy(start_conf)

                ind_a = 0
                for a in self.moved_atoms[c]:

                    if self.test_model == 'linear':
                        conf[a,:] += [self.displacement,self.displacement,self.displacement]

                    if self.test_model == 'non-linear':

                        if ind_a==0:
                            conf[a,:] += [c*self.displacement,\
                                          0,\
                                          self.displacement-c*self.displacement]
                        else:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            radius = np.sqrt(np.sum((conf[a,0:2]-conf[a_minus_1,0:2])**2))
                            angle_of_rotation = (-1)**(ind_a%2)*self.displacement/radius
                            x = np.cos(angle_of_rotation)*(conf[a,0]-conf[a_minus_1,0])-np.sin(angle_of_rotation)*(conf[a,1]-conf[a_minus_1,1])
                            y = np.sin(angle_of_rotation)*(conf[a,0]-conf[a_minus_1,0])+np.cos(angle_of_rotation)*(conf[a,1]-conf[a_minus_1,1])
                            conf[a,0:2] = conf[a_minus_1,0:2]+[x,y]

                            radius = np.sqrt(np.sum((conf[a,1:3]-conf[a_minus_1,1:3])**2))
                            angle_of_rotation = (-1)**(c)*0.5*self.displacement/radius
                            y = np.cos(angle_of_rotation)*(conf[a,1]-conf[a_minus_1,1])-np.sin(angle_of_rotation)*(conf[a,2]-conf[a_minus_1,2])
                            z = np.sin(angle_of_rotation)*(conf[a,1]-conf[a_minus_1,1])+np.cos(angle_of_rotation)*(conf[a,2]-conf[a_minus_1,2])
                            conf[a,1:3] = conf[a_minus_1,1:3]+[y,z]

                    '''
                        if ind_a%4==0:
                            conf[a,:] += [c*self.displacement,\
                                          0,\
                                          self.displacement-c*self.displacement]
                        if ind_a%4==1:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            conf[a,:] += [self.displacement*np.sin(conf[a_minus_1,0]**2),\
                                          self.displacement*np.cos(conf[a_minus_1,0]**2),\
                                          self.displacement*np.sin(conf[a_minus_1,2]**2)]
                            print([self.displacement*np.sin(conf[a_minus_1,0]**2),\
                                          self.displacement*np.cos(conf[a_minus_1,0]**2),\
                                          self.displacement*np.sin(conf[a_minus_1,2]**2)])
                        if ind_a%4==2:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            a_minus_2 = self.moved_atoms[c][ind_a-2]
                            conf[a,:] += [self.displacement*np.cos(np.exp(conf[a_minus_1,0])),\
                                          self.displacement*np.cos(np.exp(conf[a_minus_2,0])),\
                                          -self.displacement*np.sin(np.exp(conf[a_minus_1,2])+np.exp(conf[a_minus_2,2]))]
                            print([self.displacement*np.cos(np.exp(conf[a_minus_1,0])),\
                                          self.displacement*np.cos(np.exp(conf[a_minus_2,0])),\
                                          -self.displacement*np.sin(np.exp(conf[a_minus_1,2])+np.exp(conf[a_minus_2,2]))])
                        if ind_a%4==3:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            a_minus_2 = self.moved_atoms[c][ind_a-2]
                            conf[a,:] += [self.displacement*(np.sin(conf[a_minus_1,0]*conf[a_minus_2,0])-np.cos(np.sum(conf[a,:]))),\
                                          self.displacement*(np.cos(conf[a_minus_1,0]+conf[a_minus_2,0])),\
                                          self.displacement*(np.cos(conf[a_minus_2,2]**2))]
                            print([self.displacement*(np.sin(conf[a_minus_1,0]*conf[a_minus_2,0])-np.cos(np.sum(conf[a,:]))),\
                                          self.displacement*(np.cos(conf[a_minus_1,0]+conf[a_minus_2,0])),\
                                          self.displacement*(np.cos(conf[a_minus_2,2]**2))])
                        '''

                    if self.test_model == 'non-linear-random-displacement':
                        if ind_a%4==0:
                            conf[a,:] += [c*self.displacement+np.random.rand()*self.displacement,\
                                          0+np.random.rand()*self.displacement,\
                                          self.displacement-c*self.displacement+np.random.rand()*self.displacement]
                        if ind_a%4==1:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            conf[a,:] += [self.displacement-c*self.displacement,\
                                          self.displacement*np.sin(conf[a_minus_1,1]**2),\
                                          c*self.displacement]
                        if ind_a%4==2:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            a_minus_2 = self.moved_atoms[c][ind_a-2]
                            conf[a,:] += [self.displacement*np.cos(np.exp(conf[a_minus_1,1])),\
                                          self.displacement,\
                                          -self.displacement*np.sin(np.exp(conf[a_minus_2,2]))]
                        if ind_a%4==3:
                            a_minus_1 = self.moved_atoms[c][ind_a-1]
                            a_minus_2 = self.moved_atoms[c][ind_a-2]
                            conf[a,:] += [self.displacement*(np.sin(conf[a_minus_1,0]*conf[a_minus_2,0])-np.cos(np.sum(conf[a,:]))),\
                                          self.displacement+c*self.displacement,\
                                          0]

                    if self.test_model == 'non-linear-p-displacement':
                        decision = np.random.rand()
                        if decision>=0.5:
                            if ind_a%4==0:
                                conf[a,:] += [c*self.displacement,\
                                              0,\
                                              self.displacement-c*self.displacement]
                            if ind_a%4==1:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                conf[a,:] += [self.displacement-c*self.displacement,\
                                              self.displacement*np.sin(conf[a_minus_1,1]**2),\
                                              c*self.displacement]
                            if ind_a%4==2:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                a_minus_2 = self.moved_atoms[c][ind_a-2]
                                conf[a,:] += [self.displacement*np.cos(np.exp(conf[a_minus_1,1])),\
                                              self.displacement,\
                                              -self.displacement*np.sin(np.exp(conf[a_minus_2,2]))]
                            if ind_a%4==3:
                                a_minus_1 = self.moved_atoms[c][ind_a-1]
                                a_minus_2 = self.moved_atoms[c][ind_a-2]
                                conf[a,:] += [self.displacement*(np.sin(conf[a_minus_1,0]*conf[a_minus_2,0])-np.cos(np.sum(conf[a,:]))),\
                                              self.displacement+c*self.displacement,\
                                              0]

                    ind_a = ind_a+1

                if self.noise_natoms is not None:
                    for a in self.moved_noise_atoms:
                        if frame_idx%3==0:
                            conf[a,:] += [10*self.displacement,-10*self.displacement,10*self.displacement]

                conf = self._perturb(conf)
                features = self._to_features(conf)
                data[frame_idx,:] = features
                frame_idx += 1

        return data, labels

    def _generate_conformation(self):
        conf = np.zeros((self.natoms, 3))
        for n in range(self.natoms):
            conf[n] = (np.random.rand(3,)*2 -1) # Distributed between [-1,1)
        return conf

    def _perturb(self, conf):
        for n in range(self.natoms):
            conf[n] += (np.random.rand(3,)*2 - 1)*self.noise_level
        return conf

    def _to_features(self, conf):
        dists = np.empty((self.nfeatures,))
        idx = 0
        for n1, coords1 in enumerate(conf):
            for n2 in range(n1 + 1, self.natoms):
                coords2 = conf[n2]
                dists[idx] = np.linalg.norm(coords1-coords2) # Not inverse dist, is it ok?
                idx += 1
        return dists

