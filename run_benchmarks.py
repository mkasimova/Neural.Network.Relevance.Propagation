from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import glob
import argparse
from benchmarking import configuration

logger = logging.getLogger("benchmarking")


def run(extractor_type="KL",
        n_splits=1,
        n_iterations=10,
        feature_type='cartesian_rot',
        n_iter_per_example=10,
        test_model='linear',
        overwrite=False,
        output_dir="output/benchmarking/"):
    test_noise = [1e-2, 1e-2, 2e-1, 2e-1]
    for j, j_noise in enumerate(test_noise):
        for k in range(n_iter_per_example):
            modeldir = "{}/{}/{}/{}/noise_{}/iter_{}/".format(output_dir, extractor_type, feature_type, test_model,
                                                              j_noise, k)
            if os.path.exists(modeldir):
                existing_files = glob.glob(modeldir + "*.npy")
                if len(existing_files) > 0 and not overwrite:
                    logger.debug("File %s already exists. skipping computations", glob[0])
                    continue
            else:
                os.makedirs(modeldir)
            samples, cluster_indices, moved_atoms, feature_to_resids = configuration.generate_data(test_model, j,
                                                                                                   j_noise,
                                                                                                   feature_type)

            feature_extractors = configuration.create_feature_extractors(extractor_type,
                                                                         samples=samples,
                                                                         cluster_indices=cluster_indices,
                                                                         n_splits=n_splits,
                                                                         n_iterations=n_iterations)
            for i_extractor, extractor in enumerate(feature_extractors):
                extractor.extract_features()
                pp = extractor.postprocessing(predefined_relevant_residues=moved_atoms,
                                              rescale_results=True,
                                              filter_results=False,
                                              working_dir=modeldir,
                                              feature_to_resids=feature_to_resids)
                pp.average()
                pp.evaluate_performance()
                pp.persist()


def create_argparser():
    _bool_lambda = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser(
        epilog='Benchmarking for demystifying')
    parser.add_argument('--extractor_type', type=str, help='', required=True)
    parser.add_argument('--output_dir', type=str, help='', default="output/benchmarking/")
    parser.add_argument('--test_model', type=str, help='', default="linear")
    parser.add_argument('--feature_type', type=str, help='', default="cartesian_rot")
    parser.add_argument('--overwrite', type=_bool_lambda, help='', default=False)
    return parser


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    run(extractor_type=args.extractor_type,
        output_dir=args.output_dir,
        feature_type=args.feature_type,
        overwrite=args.overwrite,
        test_model=args.test_model)
