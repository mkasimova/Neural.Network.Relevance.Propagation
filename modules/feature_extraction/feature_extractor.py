import sys
import numpy as np
import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import modules.utils as utils
from sklearn.model_selection import KFold
from scipy.spatial.distance import squareform

logger = logging.getLogger("Extracting feature")

class FeatureExtractor(object):
	
	def __init__(self,samples, labels=None, scaling=True, n_splits=20, n_iterations=3, name=''):
		# Setting parameters
		self.samples = samples
		self.labels = labels
		self.n_splits = n_splits
		self.n_iterations = n_iterations
		self.scaling = scaling
		return
	
	def split_train_test(self):
		"""
		Split the data into n_splits training and test sets
		"""
		kf = KFold(n_splits=self.n_splits, shuffle=False)
		
		train_inds = []
		test_inds = []
		
		for train_ind, test_ind in kf.split(self.samples):
			train_inds.append(train_ind)
			test_inds.append(test_ind)
		return train_inds, test_inds

	def get_train_test_set(self, train_ind, test_ind):
		"""
		Get the train and test set given their sample/label indices.
		"""
		train_set = self.samples[train_ind,:]
		test_set = self.samples[test_ind,:]
		
		if self.labels is not None:
			test_labels = self.labels[test_ind,:]
			train_labels = self.labels[train_ind,:]
		else:
			test_labels = None
			train_labels = None
		
		return train_set, test_set, train_labels, test_labels		
		
	
	def train(self, train_set, train_labels):
		pass
	
	def get_feature_importance(self, model, samples, labels):
		pass
	
	def extract_features(self):
		
		train_inds, test_inds = self.split_train_test()
		errors = np.zeros(self.n_splits*self.n_iterations)

		feats = []
		
		for i_split in range(self.n_splits):
			for i_iter in range(self.n_iterations):
				
				logger.info("Iteration %s of %s", i_split*self.n_iterations+i_iter+1, self.n_splits*self.n_iterations)
				
				train_set, test_set, train_labels, test_labels = \
									self.get_train_test_set(train_inds[i_split],\
									test_inds[i_split])
				
				if self.scaling:
					train_set, perc_2, perc_98, scaler = utils.scale(train_set)
					
					test_set, perc_2, perc_98, scaler = utils.scale(test_set,\
		                                                 perc_2, perc_98,scaler)
				
				# Train model
				model = self.train(train_set, train_labels)
				
				if self.labels is not None and model is not None:
					# Test classifier
					error = utils.check_for_overfit(test_set, test_labels, model)
					errors[i_split*self.n_iterations + i_iter] = error
					
					logger.info("Error: %s",errors[i_split*self.n_iterations + i_iter])
				
				if errors[i_split*self.n_iterations + i_iter] < 5:
					logger.info("Error below 5% - computing feature importance.")
					# Get feature importances
					feature_importance = self.get_feature_importance(model, train_set, train_labels)
					feats.append(feature_importance)
		
		feats = np.asarray(feats)
		
		feats_std = np.std(feats,axis=0)
		feats = np.mean(feats,axis=0)
		
		return feats, feats_std, errors

