import argparse
import os
import sys

python_path = os.path.dirname(__file__)
sys.path.append(python_path + '/modules/')

from modules import utils, feature_extraction as fe, postprocessing as pp, visualization
from modules import filtering, data_projection as dp

import numpy as np
import comparison_bw_fe as comp_fe
from modules import relevance_propagation as relprop

def main(parser):

	# Known important residues
	common_peaks = [109,144,124,145,128,105,112,136,108,141,92]

	shuffle_data = True

	args = parser.parse_args()
	working_dir = args.out_directory
	n_runs = args.number_of_runs
	samples = np.load(args.feature_list)

	cluster_indices = np.loadtxt(args.cluster_indices)

	# Remove samples with negative cluster labels (such points are transition points)
	samples = samples[cluster_indices>=0]
	cluster_indices = cluster_indices[cluster_indices>=0]

	# Shift cluster indices to start at 0
	cluster_indices -= cluster_indices.min()

	if shuffle_data:
		# Permute blocks of 100 frames
		n_samples = samples.shape[0]
		n_samples = int(n_samples/100)*100
		inds = np.arange(n_samples)
		inds = inds.reshape((int(n_samples/100),100))
		perm_inds = np.random.permutation(inds)
		perm_inds = np.ravel(perm_inds)

		samples = samples[perm_inds]
		cluster_indices = cluster_indices[perm_inds]

	pdb_file = args.pdb_file

	labels = cluster_indices

	lower_distance_cutoff = 1.0
	upper_distance_cutoff = 1.0
	n_components = 20

	# Check if samples format is correct
	if len(samples.shape)!=2:
		sys.exit("Matrix with features should have 2 dimensions")

	kwargs = {'samples': samples, 'cluster_indices': labels,
			  'filter_by_distance_cutoff': True, 'lower_bound_distance_cutoff': lower_distance_cutoff,
			  'upper_bound_distance_cutoff': upper_distance_cutoff, 'use_inverse_distances': True,
					'n_splits': args.number_of_k_splits, 'n_iterations': args.number_of_iterations, 'scaling': True}


	feature_extractors = [
		fe.PCAFeatureExtractor(variance_cutoff=0.75, **kwargs),
		fe.RbmFeatureExtractor(relevance_method="from_components", **kwargs),
		fe.MlpAeFeatureExtractor(activation=relprop.relu, classifier_kwargs={
			'solver':'adam',
			'hidden_layer_sizes':(100,)
		}, **kwargs),
		fe.RandomForestFeatureExtractor(one_vs_rest=True,classifier_kwargs={'n_estimators':500}, **kwargs),
		fe.KLFeatureExtractor(**kwargs),
		fe.MlpFeatureExtractor(classifier_kwargs={'hidden_layer_sizes':(120,),
												  'solver':'adam',
												  'max_iter':1000000
												  },activation=relprop.relu,**kwargs),
	]
	
	postprocessors = []
	for extractor in feature_extractors:

		tmp_pp = []
		for i_run in range(n_runs):
			extractor.extract_features()
			# Post-process data (rescale and filter feature importances)
			p = extractor.postprocessing(working_dir=working_dir, rescale_results=True,
								 filter_results=False, feature_to_resids=None, pdb_file=pdb_file)
			p.average().evaluate_performance()
			p.persist()

			# Add common peaks
			tmp_pp.append(p)

		postprocessors.append(tmp_pp)

	visualization.visualize(postprocessors, show_projected_data=False, highlighted_residues=common_peaks)

parser = argparse.ArgumentParser(epilog='Feature importance extraction.')
parser.add_argument('-od', '--out_directory', help='Folder where files are written.', default='')
parser.add_argument('-y', '--cluster_indices', help='Cluster indices.', default='')
parser.add_argument('-f', '--feature_list', help='Matrix with features [nSamples x nFeatures]', default='')
parser.add_argument('-n_iter', '--number_of_iterations', help='Number of iterations to average each k-split over.',type=int, default=10)
parser.add_argument('-n_runs', '--number_of_runs', help='Number of iterations to average performance on.',type=int, default=3)
parser.add_argument('-n_splits', '--number_of_k_splits', help='Number of k splits in K-fold cross-validation.',type=int, default=10)
parser.add_argument('-pdb', '--pdb_file', help='PDB file to which the results will be mapped.', default=None)

main(parser)
