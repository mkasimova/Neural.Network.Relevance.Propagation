import argparse
import sys
import os
import shutil
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import modules
from modules.feature_extraction.feature_extractor import FeatureExtractor
import logging
logging.basicConfig(
	stream=sys.stdout,
	level=logging.DEBUG,
	format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
	datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("RF featurizer")

class RF_feature_extract(FeatureExtractor):
	
	def __init__(self,samples, labels, n_splits=10, scaling=True, n_estimators=20, njobs=4, n_iterations=3):
		FeatureExtractor.__init__(self, samples, labels, n_iterations=n_iterations, n_splits=n_splits, scaling=scaling, name="RF")
		self.n_estimators = n_estimators
		self.njobs = njobs
		return
	
	def train(self, train_set, train_labels):
		logger.info('Training RF')
		# Construct and train classifier
		classifier = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.njobs)
		classifier.fit(train_set, train_labels)
		return classifier
	
	def get_feature_importance(self, classifier, data, labels):
		return classifier.feature_importances_
	

