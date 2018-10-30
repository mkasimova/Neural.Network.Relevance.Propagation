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
    
    def __init__(self, natoms, nclusters, nframes_per_cluster, noise_level=1.0):
        """
        Class which generates artificial atoms, puts them into artifical clusters and adds noise to them
        :param natoms: number of atoms
        :param nclusters: number of clusters
        :param nframes_per_cluster:
        :param noise_leve: strength of noise to be added
        """
        self.natoms = natoms
        self.nclusters = nclusters
        self.nframes_per_cluster = nframes_per_cluster
        self.noise_level = noise_level
        
    def generate_clusters(self):
        """
        Generates actual cluster centers
        """
        pass
    
    def generate_frames(self):
        """
        Generates and returns artificial frames with noise for the clusters
        """
        pass
        
