import argparse
import sys
import os
import shutil
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import squareform

class RF_feature_extract(object):
	
	n_iterations = 10

	samples = []
	labels = []
	
	def __init__(self,samples, labels, n_iterations=10):
		
		# Setting parameters
		self.n_iterations = n_iterations
		
		# Reading data
		self.samples = samples
		self.labels = labels
		return
	
	def train_RF(self,n_estimators=100,njobs=4):
		print('Training RF')
		# Construct and train classifier
		self.classifier = RandomForestClassifier(n_estimators=n_estimators, n_jobs=njobs)
		self.classifier.fit(self.samples, self.labels)
		return

	def extract_features(self):
		
		for i_iter in range(self.n_iterations):
			self.train_RF()
			# Get feature importances
			feature_importance = self.classifier.feature_importances_
			summed_FI = np.sum(squareform(feature_importance),axis=1)
			
			if i_iter == 0:
				summed_feats = np.zeros((self.n_iterations,summed_FI.shape[0]))
				feats = np.zeros((self.n_iterations,feature_importance.shape[0]))
			
			summed_feats[i_iter,:] = summed_FI
			feats[i_iter,:] = feature_importance
		
		return feats, summed_feats

