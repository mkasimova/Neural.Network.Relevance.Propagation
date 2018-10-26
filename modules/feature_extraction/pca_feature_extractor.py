import argparse
import sys
import os
import shutil
import time
import numpy as np
from sklearn.decomposition import PCA

from modules.feature_extraction.feature_extractor import FeatureExtractor
import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("PCA featurizer")

class PCA_feature_extract(FeatureExtractor):
	
	def __init__(self,samples, labels=None, n_components=None, n_splits=10, n_iterations=3, scaling=False):
		FeatureExtractor.__init__(self, samples, labels, n_splits=n_splits, scaling=scaling, name="PCA")
		self.n_components = n_components
		return
	
	def train(self, train_set, train_labels):
		logger.info('Training PCA')
		# Construct and train PCA
		model = PCA(n_components=self.n_components)
		model.fit(train_set)
		return model
	
	def get_n_components(self, model):
		""" 
		Decide the number of components to keep based on a 90% variance cutoff
		"""
		explained_var = model.explained_variance_ratio_
		n_components = 1
		total_var_explained = explained_var[0]
		for i in range(1,explained_var.shape[0]):
			if total_var_explained + explained_var[i] < 0.9:
				total_var_explained += explained_var[i]
				n_components += 1
		logger.info('Selecting %s components',n_components)
		return n_components
	
	def get_feature_importance(self, model):
		
		n_components = self.n_components
		if (self.n_components is None) or (self.n_components > 1):
			n_components = self.get_n_components(model)
		
		feature_importances = np.sum(model.components_[0:n_components],axis=0)
		return feature_importances
	
