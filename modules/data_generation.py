from __future__ import absolute_import, division, print_function
import logging
import sys
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("DataGenerator")


class DataGenerator(object):


    def __init__(self, natoms, nclusters, natoms_per_cluster, nframes_per_cluster, test_model='linear', noise_level=1e-2, noise_natoms=None, displacement=0.1, feature_type='inv-dist'):
        """
        Class which generates artificial atoms, puts them into artifical clusters and adds noise to them
        :param natoms: number of atoms
        :param nclusters: number of clusters
        :param nframes_per_cluster:
        :param test_model: 'linear','non-linear','non-linear-random-displacement','non-linear-p-displacement'
        :param noise_level: strength of noise to be added
        :param noise_natoms: number of atoms for constant noise
        :param displacement: length of displacement vector for cluster generation
        :param feature_type: 'inv-dist' to use inversed inter-atomic distances (natoms*(natoms-1)/2 features in total) or anything that starts with 'cartesian' to use atom xyz coordiantes (3*natoms features). Use 'cartesian_rot', 'cartesian_trans' or 'cartesian_rot_trans' to add a random rotation and/or translation to xyz coordaintes
        """

        if natoms < nclusters:
            raise Exception("Cannot have more clusters than atoms")
        self.natoms = natoms
        self.nclusters = nclusters
        self.natoms_per_cluster = natoms_per_cluster
        self.nframes_per_cluster = nframes_per_cluster
        self.test_model = test_model
        self.noise_level = noise_level
        self.noise_natoms = noise_natoms
        self.displacement = displacement
        self.feature_type = feature_type
        self.nfeatures = int(self.natoms*(self.natoms-1)/2) if self.feature_type == 'inv-dist' else self.natoms*3
        self.nsamples = self.nframes_per_cluster*self.nclusters


    def Generate_Data_ClustersLabels(self):
        """
        Generate data [ nsamples x nfeatures ] and clusters labels [ nsamples ]
        """
        logger.debug("Selecting atoms for clusters ...")

        moved_atoms = []

        for c in range(self.nclusters):

            # list of atoms to be moved in a selected cluster c
            moved_atoms_c = self._pick_atoms(self.natoms_per_cluster[c], moved_atoms)
            moved_atoms.append(moved_atoms_c)

        self.moved_atoms = moved_atoms

        if self.noise_natoms is not None:

            logger.debug("Selecting atoms for constant noise ...")
            self.moved_atoms_noise = self._pick_atoms(self.noise_natoms, moved_atoms)

        logger.info("Generating frames ...")
        conf0 = self._generate_conformation0()
        labels = np.zeros((self.nsamples, self.nclusters))
        data = np.zeros((self.nsamples, self.nfeatures))

        frame_idx = 0

        for f in range(self.nframes_per_cluster):

            for c in range(self.nclusters):

                labels[frame_idx, c] = 1
                conf = np.copy(conf0)

                # Move atoms in each cluster
                a_idx = 0
                for a in self.moved_atoms[c]:
                    conf[a,:] = self._move_an_atom(c, conf, a_idx, a)
                    a_idx = a_idx + 1

                # Add constant noise
                if self.noise_natoms is not None:
                    for a in self.moved_atoms_noise:
                        if frame_idx%3==0: # move noise atoms every 3rd frame
                            conf[a,:] += [ 10*self.displacement, 10*self.displacement, 10*self.displacement]

                # Add random noise
                conf = self._perturb(conf)

                # Generate features
                if self.feature_type == "inv-dist":
                    features = self._to_inv_dist(conf)
                elif self.feature_type.startswith("cartesian"):
                    features = self._to_cartesian(conf)

                data[frame_idx,:] = features
                frame_idx += 1

        return data, labels


    def _pick_atoms(self, natoms_to_pick, moved_atoms):
        """
        Select atoms to be moved for each cluster
        OR
        Select atoms to be moved for constant noise
        """

        moved_atoms_c = []

        for a in range(natoms_to_pick):

            while True:
                atom_to_move = np.random.randint(self.natoms)
                if ( atom_to_move not in moved_atoms_c ) and \
                   ( atom_to_move not in [y for x in moved_atoms for y in x] ):
                    moved_atoms_c.append(atom_to_move)
                    break

        return moved_atoms_c


    def _generate_conformation0(self):

        conf = np.zeros((self.natoms, 3))
        for n in range(self.natoms):
            conf[n] = (np.random.rand(3,)*2 -1) # Distributed between [-1,1)

        return conf


    def _move_an_atom(self, c, conf, a_idx, a):
        """
        Move an atom of a cluster
        """

        if self.test_model == 'linear':
            conf[a,:] += [ self.displacement, self.displacement, self.displacement ]

        elif self.test_model == 'non-linear':
            if a_idx == 0:
                conf[a,:] += [ c*self.displacement, 0, self.displacement-c*self.displacement ]
            else:
                conf[a,:] = self._move_an_atom_along_circle(c, conf, a_idx, a)

        elif self.test_model == 'non-linear-random-displacement':
            if a_idx == 0:
                conf[a,:] += [ c*self.displacement + np.random.rand() * self.displacement,\
                               0 + np.random.rand() * self.displacement,\
                               self.displacement - c*self.displacement + np.random.rand() * self.displacement ] # displacement of the first atom is random
            else:
                conf[a,:] = self._move_an_atom_along_circle(c, conf, a_idx, a)

        elif self.test_model == 'non-linear-p-displacement':
            decision = np.random.rand() # atoms move with [decision] probability
            if decision>=0.5:
                if a_idx == 0:
                    conf[a,:] += [ c*self.displacement, 0, self.displacement-c*self.displacement ]
                else:
                    conf[a,:] = self._move_an_atom_along_circle(c, conf, a_idx, a)

        return conf[a,:]


    def _move_an_atom_along_circle(self, c, conf, a_idx, a):
        """
        Move an atom of a cluster along the circle whose center is the previous atom (aa)
        First in XY plane
        And next in YZ plane
        """

        aa = self.moved_atoms[c][a_idx-1] # previous atom

        radius = np.sqrt(np.sum((conf[a,0:2]-conf[aa,0:2])**2))
        angle_of_rotation = (-1)**(a_idx) * self.displacement/radius # direction of rotation is defined by a_idx - atom index
        conf[a,0:2] = conf[aa,0:2] + self._rotate(angle_of_rotation, conf[a]-conf[aa], [0,1])[0:2]


        radius = np.sqrt(np.sum((conf[a,1:3]-conf[aa,1:3])**2))
        angle_of_rotation = (-1)**(c) * self.displacement/radius # direction of rotation is defined by c - cluster index
        conf[a,1:3] = conf[aa,1:3] + self._rotate(angle_of_rotation, conf[a]-conf[aa], [1,2])[1:3]

        return conf[a,:]


    def _rotate(self, phi, xyz, dims):

        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        xyz = xyz.T # to work for an N-dim array
        xy = xyz[dims]
        xyz[dims[0]] = (cos_phi*xy[0]+sin_phi*xy[1])
        xyz[dims[1]] = (-sin_phi*xy[0]+cos_phi*xy[1])
        xyz = xyz.T

        return xyz


    def _perturb(self, conf):

        for n in range(self.natoms):
            conf[n] += (np.random.rand(3,)*2 - 1)*self.noise_level

        return conf


    def _to_inv_dist(self, conf):

        feats = np.empty((self.nfeatures))
        idx = 0
        for n1, coords1 in enumerate(conf):
            for n2 in range(n1 + 1, self.natoms):
                coords2 = conf[n2]
                feats[idx] = np.linalg.norm(coords1-coords2)
                idx += 1

        return feats


    def _to_cartesian(self, conf):

        if "_rot" in self.feature_type:
            conf = self._random_rotation(conf)
        if "_trans" in self.feature_type:
            conf = self._random_translation(conf)

        feats = np.empty((self.nfeatures))
        idx = 0
        for n1, coords1 in enumerate(conf):
                feats[idx] = coords1[0] #x
                idx += 1
                feats[idx] = coords1[1] #y
                idx += 1
                feats[idx] = coords1[2] #z
                idx += 1

        return feats


    def _random_rotation(self, xyz):
        """
        Randomly rotate each frame along each axis
        """
        # Random angles between 0 and 2pi
        phi, psi, theta = 2*np.pi*np.random.rand(), 2*np.pi*np.random.rand(), np.pi*np.random.rand()
        # see http://mathworld.wolfram.com/EulerAngles.html
        xyz = self._rotate(phi, xyz, [0,1]) # rotate xy plane plane
        xyz = self._rotate(theta, xyz, [1,2]) # rotate new yz plane
        xyz = self._rotate(psi, xyz, [0,1]) # rotate new xy plane

        return xyz


    def _random_translation(self, xyz):
        """
        Randomly translate each frame along each axis ; does not support PBC
        """
        [dx, dy, dz] = 5*(np.random.rand(3) - 0.5) # random values within box size
        xyz[:, 0] += dx
        xyz[:, 1] += dy
        xyz[:, 2] += dz

        return xyz


    def feature_to_resids(self):

        if self.feature_type == 'inv-dist':
            return None # TODO fix later; default anyway
        elif self.feature_type.startswith("cartesian"):
            mapping = []
            for a in range(self.natoms):
                mapping.append([a, a]) #x
                mapping.append([a, a]) #y
                mapping.append([a, a]) #z
            return np.array(mapping)
        else:
            raise Exception("Unknown feature type {}".format(self.feature_type))

