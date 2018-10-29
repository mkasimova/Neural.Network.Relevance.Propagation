import numpy as np
from scipy.spatial.distance import squareform



def state_average(self, all_relevances, class_indices):
	"""
	Average relevance over each state/cluster. 
	"""
	if len(all_relevances)==3:
		class_labels = np.unique(class_indices)
		n_classes = class_labels.shape[0]
		
		class_relevance = np.zeros((all_relevances.shape[0],n_classes,all_relevances.shape[2]))
		
		for k in range(n_classes):
			class_relevance[:,k,:] = \
					np.mean(all_relevances[:,np.where(clustering_train==\
					class_labels[k])[0],:],axis=0)
		
		return class_relevance
	else:
		return all_relevances
	

def filter_feature_importance(self, average_relevance, std_relevance=None, n_sigma_threshold=2):
	"""
	Filter feature importances based on significance.
	Return filtered residue feature importances (average + std within the states/clusters).
	"""
	if len(average_relevance.shape) == 1:
		n_states = 1
		n_features = squareform(average_relevance[:]).shape[0]
		average_relevance = average_relevance[:,np.newaxis].T
	else:
		n_states = average_relevance.shape[0]
		n_features = squareform(average_relevance[0,:]).shape[0]
	
	residue_relevance_ave = np.zeros((n_states,n_features))
	residue_relevance_std = np.zeros((n_states,n_features))
	
	for i in range(n_states):
		ind_nonzero = np.where(average_relevance[i,:]>0)
		global_mean = np.mean(average_relevance[i,ind_nonzero])
		global_sigma = np.std(average_relevance[i,ind_nonzero])
		
		average_relavance_mat = squareform(average_relevance[i,:])
		
		# If no standard deviation present, 
		if std_relevance is not None:
			std_relavance_mat = squareform(std_relevance[i,:])
		else:
			residue_relevance_std = np.zeros(residue_relevance_ave.shape)
		
		for j in range(n_features):
			# Identify significant features			
			ind_above_sigma = np.where(average_relavance_mat[j,:]>=\
	                                  (global_mean + n_sigma_threshold*global_sigma))[0]
			
			# Sum over significant features (=> per-residue relevance)
			residue_relevance_ave[i,j] = np.sum(average_relavance_mat[j,ind_above_sigma])
			if std_relevance is not None:
				residue_relevance_std[i,j] = np.sqrt(np.sum(std_relavance_mat[j,ind_above_sigma]**2))
	
	return residue_relevance_ave, residue_relevance_std

def rescale_feature_importance(self, feature_importance):
	"""
	Min-max rescale feature importances
	"""
	if len(feature_importance.shape)==3:
		for i in range(feature_importance.shape[0]):
			for j in range(feature_importance.shape[1]):
				feature_importance[i,j,:] = (feature_importance[i,j,:]-np.min(feature_importance[i,j,:]))/\
	        		(np.max(feature_importance[i,j,:])-np.min(feature_importance[i,j,:])+1e-9)
	
	return feature_importance
