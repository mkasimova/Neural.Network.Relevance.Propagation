import argparse
import os
import sys

python_path = os.path.dirname(__file__)
sys.path.append(python_path + '/modules/')

from modules import utils, feature_extraction as fe, postprocessing as pp, visualization
from modules import filtering

import numpy as np


def main(parser):

    args = parser.parse_args()
    working_dir = args.out_directory  # TODO set appropriately
    samples = np.load(args.feature_list)
    cluster_indices = np.loadtxt(args.cluster_indices)
    pdb_file = args.pdb_file

    ''' Data pre-processing (only for HCN!!!!!) ''' # TODO remove in the final version
    samples = utils.vectorize(samples)
    points_to_keep = [[0,1450],\
                      [1750,3650],\
                      [4150,5900],\
                      [6800,-1]]
    samples, cluster_indices = filtering.keep_datapoints(samples,cluster_indices,points_to_keep)
    ''' '''

    # Check if samples format is correct
    if len(samples.shape)!=2:
        sys.exit("Matrix with features should only have 2 dimensions")

    feature_extractors = [
        fe.MlpFeatureExtractor(samples,\
                               cluster_indices,\
                               n_splits=args.number_of_k_splits,\
                               n_iterations=args.number_of_iterations,\
                               scaling=True,\
                               filter_by_distance_cutoff=False,\
                               filter_by_DKL=False,\
                               filter_by_KS_test=False,\
                               hidden_layer_sizes=(100,)),
        fe.ElmFeatureExtractor(samples,\
                               cluster_indices,\
                               n_splits=args.number_of_k_splits,\
                               n_iterations=args.number_of_iterations,\
                               scaling=True,\
                               filter_by_distance_cutoff=False,\
                               filter_by_DKL=False,\
                               filter_by_KS_test=False),
        fe.KLFeatureExtractor(samples,\
                              cluster_indices,\
                              n_splits=args.number_of_k_splits,\
                              scaling=True,\
                              filter_by_distance_cutoff=False,\
                              filter_by_DKL=False,\
                              filter_by_KS_test=False),
        fe.PCAFeatureExtractor(samples,\
                               cluster_indices,
                               n_splits=args.number_of_k_splits,\
                               scaling=False,\
                               filter_by_distance_cutoff=False,\
                               filter_by_DKL=False,\
                               filter_by_KS_test=False,\
                               n_components=1),
        fe.RandomForestFeatureExtractor(samples,\
                                        cluster_indices,\
                                        n_splits=args.number_of_k_splits,\
                                        n_iterations=args.number_of_iterations,\
                                        scaling=True,\
                                        filter_by_distance_cutoff=False,\
                                        filter_by_DKL=False,\
                                        filter_by_KS_test=False)
    ]

    postprocessors = []
    for extractor in feature_extractors:
        feats, std_feats, errors = extractor.extract_features()

        # Post-process data (rescale and filter feature importances)
        p = pp.PostProcessor(extractor,\
                             feats,\
                             std_feats,\
                             errors,\
                             cluster_indices,
                             working_dir,\
                             rescale_results=True,\
                             filter_results=True,\
                             feature_to_resids=None,\
                             pdb_file=pdb_file)
        p.average().persist()
        postprocessors.append(p)

#    visualization.visualize(postprocessors)

parser = argparse.ArgumentParser(epilog='Feature importance extraction.')
parser.add_argument('-od', '--out_directory', help='Folder where files are written.', default='')
parser.add_argument('-y', '--cluster_indices', help='Cluster indices.', default='')
parser.add_argument('-f', '--feature_list', help='Matrix with features [nSamples x nFeatures]', default='')
parser.add_argument('-n_iter', '--number_of_iterations', help='Number of iterations to average each k-split over.',type=int, default=10)
parser.add_argument('-n_splits', '--number_of_k_splits', help='Number of k splits in K-fold cross-validation.',type=int, default=10)
parser.add_argument('-pdb', '--pdb_file', help='PDB file to which the results will be mapped.', default=None)

main(parser)
