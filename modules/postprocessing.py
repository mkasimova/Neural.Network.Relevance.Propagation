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
from biopandas.pdb import PandasPdb
import modules.utils as utils
import modules.filtering as filtering
import modules.data_projection as dp

logger = logging.getLogger("postprocessing")

class PostProcessor(object):

    def __init__(self, extractor, feature_importance, std_feature_importance, test_set_errors, cluster_indices, working_dir, rescale_results=True, filter_results=False, feature_to_resids=None, pdb_file=None, predefined_relevant_residues=None):
        """
        Class which computes all the necessary averages and saves them as fields
        TODO move some functionality from class feature_extractor here
        :param extractor:
        :param feature_importance:
        :param std_feature_importance:
        :param cluster_indices:
        :param working_dir:
        :param feature_to_resids: an array of dimension nfeatures*2 which tells which two residues are involved in a feature
        """
        self.extractor = extractor
        self.feature_importance = feature_importance
        self.std_feature_importance = std_feature_importance
        self.cluster_indices = cluster_indices
        self.working_dir = working_dir
        self.pdb_file = pdb_file
        self.predefined_relevant_residues = predefined_relevant_residues

        # Rescale and filter results if needed
        self.rescale_results = rescale_results
        if rescale_results:
            self.feature_importance, self.std_feature_importance = utils.rescale_feature_importance(self.feature_importance, self.std_feature_importance)
        if filter_results:
            self.feature_importance, self.std_feature_importance = filtering.filter_feature_importance(self.feature_importance, self.std_feature_importance)

        # Put importance and std to 0 for residues pairs which were filtered out during features filtering (they are set as -1 in self.feature_importance and self.std_feature_importance)
        self.indices_filtered = np.where(self.feature_importance[:,0]==-1)[0]
        self.feature_importance[self.indices_filtered,:] = 0
        self.std_feature_importance[self.indices_filtered,:] = 0

        # Set mapping from features to residues
        self.nfeatures, self.nclusters = feature_importance.shape
        if feature_to_resids is None and self.pdb_file is None:
            feature_to_resids = utils.get_default_feature_to_resids(self.nfeatures)
        elif feature_to_resids is None and self.pdb_file is not None:
            feature_to_resids = utils.get_feature_to_resids_from_pdb(self.nfeatures,self.pdb_file)
        self.feature_to_resids = feature_to_resids

        # Set average feature importances to None
        self.importance_per_cluster = None
        self.std_importance_per_cluster = None
        self.importance_per_residue_and_cluster = None
        self.std_importance_per_residue_and_cluster = None
        self.importance_per_residue = None
        self.std_importance_per_residue = None
        self.index_to_resid = None

        # Performance metrics
        self.predefined_relevant_residues = predefined_relevant_residues
        self.average_std = None
        self.test_set_errors = test_set_errors.mean()
        self.data_projector = None
        self.tp_rate = None
        self.fp_rate = None
        self.auc = None

    def average(self):
        """
        Computes average importance per cluster and residue and residue etc.
        Sets the fields importance_per_cluster, importance_per_residue_and_cluster, importance_per_residue
        :return: itself
        """
        self.importance_per_cluster = self.feature_importance # compute_importance_per_cluster(importance, cluster_indices)
        self.std_importance_per_cluster = self.std_feature_importance
        self._compute_importance_per_residue_and_cluster()
        self._compute_importance_per_residue()

        return self

    def _compute_importance_per_residue_and_cluster(self):

        importance = self.importance_per_cluster
        index_to_resid = set(self.feature_to_resids.flatten()) # at index X we have residue number
        self.nresidues = len(index_to_resid)
        index_to_resid = [r for r in index_to_resid]
        res_id_to_index = {} # a map pointing back to the index in the array index_to_resid
        for idx, resid in enumerate(index_to_resid):
            res_id_to_index[resid] = idx
        importance_per_residue_and_cluster = np.zeros((self.nresidues, self.nclusters))
        std_importance = np.zeros((self.nresidues, self.nclusters))
        for feature_idx, rel in enumerate(importance):
            res1, res2 = self.feature_to_resids[feature_idx]
            res1 = res_id_to_index[res1]
            res2 = res_id_to_index[res2]
            importance_per_residue_and_cluster[res1, :] += rel
            importance_per_residue_and_cluster[res2, :] += rel
            std_importance[res1,:] += self.std_importance_per_cluster[feature_idx,:]**2
            std_importance[res2,:] += self.std_importance_per_cluster[feature_idx,:]**2
        std_importance = np.sqrt(std_importance)

        if self.rescale_results:
            importance_per_residue_and_cluster, std_importance = utils.rescale_feature_importance(importance_per_residue_and_cluster, std_importance)

        self.importance_per_residue_and_cluster = importance_per_residue_and_cluster
        self.std_importance_per_residue_and_cluster = std_importance
        self.index_to_resid = index_to_resid

    def _compute_importance_per_residue(self):

        importance_per_residue = self.importance_per_residue_and_cluster.mean(axis=1)
        std_importance_per_residue = np.sqrt(np.mean(self.std_importance_per_residue_and_cluster**2,axis=1))

        if self.rescale_results:
            # Adds a second axis to feed to utils.rescale_feature_importance
            importance_per_residue = importance_per_residue.reshape((importance_per_residue.shape[0],1))
            std_importance_per_residue = std_importance_per_residue.reshape((std_importance_per_residue.shape[0],1))
            importance_per_residue, std_importance_per_residue = utils.rescale_feature_importance(importance_per_residue, std_importance_per_residue)
            importance_per_residue = importance_per_residue[:,0]
            std_importance_per_residue = std_importance_per_residue[:,0]

        self.importance_per_residue = importance_per_residue
        self.std_importance_per_residue = std_importance_per_residue

    def evaluate_performance(self):
        """
        Computes -average of standard deviation (per residue)
                 -projection classification entropy
                 -area under ROC (for toy model only)
        """

        self._compute_average_std()
        #self._compute_projection_classification_entropy()

        if self.predefined_relevant_residues is not None:
            self._compute_area_under_ROC()

        return self

    def _compute_average_std(self):
        """
        Computes average standard deviation
        """
        self.average_std = self.std_importance_per_residue.mean()

        return self

    def _compute_projection_classification_entropy(self):
        """
        Computes separation of clusters in the projected space given by the feature importances
        """
        self.data_projector = dp.DataProjector(self.extractor.samples,self.cluster_indices)
        self.data_projector.project(self.importance_per_cluster).score_projection()

        return self

    def _compute_area_under_ROC(self):
        """
        Computes ROC curve and area under it
        """
        n_residues = self.importance_per_residue.shape[0]
        actives = np.chararray(n_residues)
        actives[:] = 'd'
        ind_a = [y for x in self.predefined_relevant_residues for y in x]
        actives[ind_a] = 'a'

        actives_len = len(ind_a)
        decoys_len = n_residues - actives_len

        ind_scores_sorted = np.argsort(-self.importance_per_residue)
        actives_sorted = actives[ind_scores_sorted]

        tp=0
        fp=0
        tp_rate = []
        fp_rate = []
        for i in actives_sorted:
            if i=='a':
                tp+=1
            else:
                fp+=1
            tp_rate.append(float(tp)/float(actives_len))
            fp_rate.append(float(fp)/float(decoys_len))

        auc = 0
        for i in range(len(fp_rate)-1):
            auc += (fp_rate[i+1]-fp_rate[i])*(tp_rate[i+1]+tp_rate[i])/2

        self.tp_rate = tp_rate
        self.fp_rate = fp_rate
        self.auc = auc

    def persist(self):
        """
        Save .npy files of the different averages and pdb files with the beta column set to importance
        :return: itself
        """
        directory = self.working_dir + "/{}/".format(self.extractor.name)

        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(directory + "importance_per_cluster", self.importance_per_cluster)
        np.save(directory + "importance_per_residue_and_cluster", self.importance_per_residue_and_cluster)
        np.save(directory + "importance_per_residue", self.importance_per_residue)
        np.save(directory + "feature_importance", self.feature_importance)
        np.save(directory + "std_feature_importance", self.std_feature_importance)

        if self.pdb_file is not None:
            pdb = PandasPdb()
            pdb.read_pdb(self.pdb_file)
            self._save_to_pdb(pdb, directory + "average_importance.pdb", self._map_to_correct_residues(self.importance_per_residue))
            for cluster_idx, importance in enumerate(self.importance_per_residue_and_cluster.T):
                self._save_to_pdb(pdb, directory + "cluster_{}_importance.pdb".format(cluster_idx), self._map_to_correct_residues(importance))

        return self

    def _map_to_correct_residues(self, importance_per_residue):
        """
        Maps importances to correct residue numbers
        """
        residue_to_importance = {}
        for idx, rel in enumerate(importance_per_residue):
            resSeq = self.index_to_resid[idx]
            residue_to_importance[resSeq] = rel

        return residue_to_importance

    def _save_to_pdb(self, pdb, out_file, residue_to_importance):
        """
        Saves importances into beta column of pdb file
        """
        atom = pdb.df['ATOM']
        missing_residues = []
        for i, line in atom.iterrows():
            resSeq = int(line['residue_number'])
            importance = residue_to_importance.get(resSeq, None)
            if importance is None:
                missing_residues.append(resSeq)
                importance = 0
            atom.set_value(i, 'b_factor', importance)
        if len(missing_residues) > 0:
            logger.warn("importance is None for residues %s", set(missing_residues))
        pdb.to_pdb(path=out_file, records=None, gz=False, append_newline=True)

        return self
