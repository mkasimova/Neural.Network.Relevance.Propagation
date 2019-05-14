from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib as mpl

mpl.use('Agg')  # TO DISABLE GUI. USEFUL WHEN RUNNING ON CLUSTER WITHOUT X SERVER
import argparse
import numpy as np
from benchmarking import computing, visualization, utils

logger = logging.getLogger("benchmarking")


def _fix_extractor_type(extractor_types):
    if len(extractor_types) == 1:
        et = extractor_types[0]
        if et == "supervised":
            return ["KL", "RF", "MLP"]
        elif et == "unsupervised":
            return ["PCA", "RBM", "AE"]
        elif et == "all":
            return ["KL", "RF", "MLP", "PCA", "RBM", "AE"]
    return extractor_types


def create_argparser():
    _bool_lambda = lambda x: (str(x).lower() == 'true')
    parser = argparse.ArgumentParser(
        epilog='Benchmarking for demystifying')
    parser.add_argument('--extractor_type', nargs='+', type=str, required=True)
    parser.add_argument('--output_dir', type=str, help='', default="output/benchmarking/")
    parser.add_argument('--test_model', type=str, help='', default="linear")
    parser.add_argument('--feature_type', type=str, help='', default="cartesian_rot")
    parser.add_argument('--noise_level', type=float, help='', default=1e-2)
    parser.add_argument('--displacement', type=float, help='', default=1e-1)
    parser.add_argument('--overwrite', type=_bool_lambda, help='', default=False)
    parser.add_argument('--visualize', type=_bool_lambda, help='', default=True)
    return parser


def run(args):
    extractor_types = utils._fix_extractor_type(args.extractor_type)
    visualize = args.visualize
    output_dir = args.output_dir
    best_processors = []
    feature_type = args.feature_type
    test_model = args.test_model
    noise_level = args.noise_level
    fig_filename = "{feature_type}_{test_model}_{noise_level}noise.svg".format(
        feature_type=feature_type,
        test_model=test_model,
        noise_level=noise_level)
    for et in extractor_types:
        postprocessors = computing.compute(extractor_type=et,
                                           output_dir=output_dir,
                                           feature_type=feature_type,
                                           overwrite=args.overwrite,
                                           noise_level=args.noise_level,
                                           test_model=args.test_model)
        if visualize:
            visualization.show_all(postprocessors=postprocessors,
                                   extractor_type=et,
                                   filename=fig_filename,
                                   output_dir=output_dir)
        best_processors.append(utils.find_best(postprocessors))
    if visualize:
        visualization.show_best(np.array(best_processors),
                                extractor_types,
                                filename=fig_filename,
                                output_dir=output_dir)


if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    run(args)
    logger.info("Done!")
