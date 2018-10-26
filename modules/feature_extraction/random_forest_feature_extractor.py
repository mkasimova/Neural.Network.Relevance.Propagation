import argparse
import sys
import os
import shutil
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import modules
from modules.feature_extraction.feature_extractor import FeatureExtractor

class RF_feature_extract(FeatureExtractor):
	
	def __init__(self,samples, labels, n_splits=10, scaling=True, n_estimators=20, njobs=4):
		FE.feature_extractor.__init__(self, samples, labels, n_splits=n_splits, scaling=scaling)
		self.n_estimators = n_estimators
		self.njobs = njobs
		return
	
	def train(self, train_set, train_labels):
		print('Training RF')
		# Construct and train classifier
		classifier = RandomForestClassifier(n_estimators=self.n_estimators, n_jobs=self.njobs)
		classifier.fit(train_set, train_labels)
		return classifier
	
	def get_feature_importance(self, classifier, data, labels):
		return classifier.feature_importances_
	

