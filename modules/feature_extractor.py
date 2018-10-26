import numpy as np
import Compute_Relevance as CR
from sklearn.model_selection import KFold
from scipy.spatial.distance import squareform

class feature_extractor(object):
	
	def __init__(self,samples, labels, n_iterations=10, scaling=True, n_splits=20):
		
		# Setting parameters
		self.n_iterations = n_iterations
		
		# Reading data
		self.samples = samples
		self.labels = labels
		self.n_splits = n_splits
		self.scaling = scaling
		return
	
	def split_train_test(self):
		"""
		Split the data into n_splits training and test sets
		"""
		kf = KFold(n_splits=self.n_splits, shuffle=False)
		
		train_inds = []
		test_inds = []
		
		for train_ind, test_ind in kf.split(self.samples, self.labels):
			train_inds.append(train_ind)
			test_inds.append(test_ind)
		
		return train_inds, test_inds
	
	def train(self, train_set, train_labels):
		pass

	def get_feature_importance(self, classifier):
		pass

	def summed_feature_importance(self,feature_importance):
		"""
		Get residue feature importance
		"""
		return np.sum(squareform(feature_importance),axis=1)
	
	def extract_features(self):
		
		train_inds, test_inds = self.split_train_test()

		for i_iter in range(self.n_splits):
			
			train_set = self.samples[train_inds[i_iter],:]
			train_labels = self.labels[train_inds[i_iter],:]
			
			test_set = self.samples[test_inds[i_iter],:]
			test_labels = self.labels[test_inds[i_iter],:]
			
			if self.scaling:
				train_set, perc_2, perc_98, scaler = CR.scale(train_set)
				
				test_set, perc_2, perc_98, scaler = scale(test_set,\
                                                                  perc_2,\
                                                                  perc_98,\
                  		                                          scaler)
			# Train classfier
			classifier = self.train(train_set,train_labels)
			
			# Test classifier
			error = CR.check_for_overfit(test_set, test_labels, classifier)
			
			# Get feature importances
			feature_importance = self.get_feature_importance(classifier)
			summed_FI = self.summed_feature_importance(feature_importance)
			
			if i_iter == 0:
				summed_feats = np.zeros((self.n_iterations,summed_FI.shape[0]))
				feats = np.zeros((self.n_iterations,feature_importance.shape[0]))
			
			summed_feats[i_iter,:] = summed_FI
			feats[i_iter,:] = feature_importance
		
		return feats, summed_feats

