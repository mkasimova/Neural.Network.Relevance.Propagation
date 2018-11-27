import argparse
import os
import sys

python_path = os.path.dirname(__file__)
sys.path.append(python_path + '/modules/')

from modules import utils, feature_extraction as fe, postprocessing as pp, visualization
from modules import filtering, data_projection as dp

import numpy as np
import comparison_bw_fe as comp_fe


def main(parser):

    n_runs = 5

    args = parser.parse_args()
    working_dir = args.out_directory
    samples = np.load(args.feature_list)
    cluster_indices = np.loadtxt(args.cluster_indices)
    pdb_file = args.pdb_file

    labels = cluster_indices

    contact_cutoff = 1.0

    # Check if samples format is correct
    if len(samples.shape)!=2:
        sys.exit("Matrix with features should only have 2 dimensions")

    feature_extractors = [
        fe.PCAFeatureExtractor(samples, labels,
                               n_splits=args.number_of_k_splits,
                               scaling=False, filter_by_distance_cutoff=True, contact_cutoff=contact_cutoff,
                               filter_by_DKL=False, filter_by_KS_test=False, n_components=None),
        #fe.RbmFeatureExtractor(samples, labels, n_splits=args.number_of_k_splits, \
        #                       n_iterations=args.number_of_iterations, scaling=True, \
        #                       filter_by_distance_cutoff=True, contact_cutoff=contact_cutoff, filter_by_DKL=False,
        #                       filter_by_KS_test=False),
        fe.RandomForestFeatureExtractor(samples, labels, n_splits=args.number_of_k_splits,
                                        n_iterations=args.number_of_iterations,
                                        scaling=True, filter_by_distance_cutoff=True, contact_cutoff=contact_cutoff,
                                        filter_by_DKL=False, filter_by_KS_test=False),
        fe.KLFeatureExtractor(samples, labels, n_splits=args.number_of_k_splits,  scaling=True,
                              filter_by_distance_cutoff=True, contact_cutoff=contact_cutoff,
                              filter_by_DKL=False, filter_by_KS_test=False),
        #fe.ElmFeatureExtractor(samples, labels,\
        #					   n_splits=args.number_of_k_splits,\
        #					   n_iterations=args.number_of_iterations,\
        #					   scaling=True,\
        #					   filter_by_distance_cutoff=True,contact_cutoff=0.8,\
        #					   filter_by_DKL=False,\
        #					  filter_by_KS_test=False),
        fe.MlpFeatureExtractor(samples, labels, n_splits=args.number_of_k_splits,
                               n_iterations=args.number_of_iterations, scaling=True,
                               filter_by_distance_cutoff=True, contact_cutoff=contact_cutoff,
                               filter_by_DKL=False, filter_by_KS_test=False,
                               hidden_layer_sizes=(100,)),
    ]


    postprocessors = []
    data_projectors = []

    for extractor in feature_extractors:
        tmp_pp = []
        tmp_dp = []
        for i_run in range(n_runs):
            feats, std_feats, errors = extractor.extract_features()

            # Post-process data (rescale and filter feature importances)
            p = pp.PostProcessor(extractor, feats, std_feats, errors,\
                                 cluster_indices, working_dir, rescale_results=True,\
                                 filter_results=False, filter_results_by_cutoff=False,\
                                 feature_to_resids=None, pdb_file=pdb_file)


            p.average().persist()

            projector = dp.DataProjector(p, samples)
            projector.project().score_projection()

            tmp_dp.append(projector)
            tmp_pp.append(p)

        postprocessors.append(tmp_pp)
        data_projectors.append(tmp_dp)

    visualization.visualize(postprocessors,data_projectors)

parser = argparse.ArgumentParser(epilog='Feature importance extraction.')
parser.add_argument('-od', '--out_directory', help='Folder where files are written.', default='')
parser.add_argument('-y', '--cluster_indices', help='Cluster indices.', default='')
parser.add_argument('-f', '--feature_list', help='Matrix with features [nSamples x nFeatures]', default='')
parser.add_argument('-n_iter', '--number_of_iterations', help='Number of iterations to average each k-split over.',type=int, default=10)
parser.add_argument('-n_splits', '--number_of_k_splits', help='Number of k splits in K-fold cross-validation.',type=int, default=10)
parser.add_argument('-pdb', '--pdb_file', help='PDB file to which the results will be mapped.', default=None)

main(parser)
