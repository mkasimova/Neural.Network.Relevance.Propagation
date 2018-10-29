import argparse
import os
import sys

python_path = os.path.dirname(__file__)
sys.path.append(python_path + '/modules/')

from modules import utils, feature_extraction as fe, postprocessing as pp, visualization

import numpy as np


def main(parser):
    args = parser.parse_args()
    samples = np.load(args.feature_list)
    cluster_indices = np.loadtxt(args.cluster_indices)
    labels = utils.create_class_labels(cluster_indices)
    working_dir = args.out_directory  # TODO set appropriately
    feature_extractors = [
        # fe.MlpFeatureExtractor(samples, labels, n_splits=4, scaling=True, hidden_layer_sizes=(100,)),
        # fe.ElmFeatureExtractor(samples, labels),
        fe.KLFeatureExtractor(samples, labels, n_splits=args.number_of_k_splits),
        fe.PCAFeatureExtractor(samples, n_components=1, n_splits=args.number_of_k_splits),
        fe.RandomForestFeatureExtractor(samples, labels, n_splits=args.number_of_k_splits,
                                        n_iterations=args.number_of_iterations)
    ]

    postprocessors = []
    for extractor in feature_extractors:
        feats, std_feats, errors = extractor.extract_features()
        # Post-process data (rescale and filter feature importances)
        p = pp.PostProcessor(extractor, feats, std_feats, cluster_indices,
                             working_dir, feature_to_resids=None)
        p.average().persist()
        postprocessors.append(p)

        """ OLD VISUALIZATION CODE
        plt.figure(1)
        for i in range(relevance.shape[1]):
            plt.plot(relevance[:, i])
            plt.plot(relevance[:, i]+std_relevance[:, i])
            plt.plot(relevance[:, i]-std_relevance[:, i])
        plt.show()
        results.append((extractor, relevance, std_relevance))
        np.save(args.out_directory + 'relevance_results' + args.file_end_name + '.npy', results)
        """
    visualization.visualize(postprocessors)


parser = argparse.ArgumentParser(epilog='Random forest classifier for feature importance extraction.')
parser.add_argument('-fe', '--file_end_name', help='End file label.', default='')
parser.add_argument('-od', '--out_directory', help='Folder where files are written.', default='')
parser.add_argument('-y', '--cluster_indices', help='Cluster indices.', default='')
parser.add_argument('-f', '--feature_list', help='Matrix with features [nSamples x nFeatures]', default='')
parser.add_argument('-n_iter', '--number_of_iterations', help='Number of iterations to average each k-split over.',
                    type=int, default=3)
parser.add_argument('-n_splits', '--number_of_k_splits', help='Number of k splits in K-fold cross-validation.',
                    type=int, default=10)
parser.add_argument('-scale', '--scale_data', help='Flag for scaling data.', action='store_true')

main(parser)
