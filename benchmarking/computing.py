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
import numpy as np
from . import configuration

logger = logging.getLogger("ex_benchmarking")


def compute(extractor_type,
            n_splits=1,
            n_iterations=10,
            feature_type='cartesian_rot',
            iterations_per_model=10,
            test_model='linear',
            overwrite=False,
            noise_level=1e-2,  # [1e-2, 1e-2, 2e-1, 2e-1],
            output_dir="output/benchmarking/"):
    """

    :param extractor_type:
    :param n_splits:
    :param n_iterations:
    :param feature_type:
    :param iterations_per_model:
    :param test_model:
    :param overwrite:
    :param noise_levels:
    :param output_dir:
    :return: postprocessors (np.array of dim iterations_per_model, nfeature_extractors)
    """
    all_postprocessors = []
    for iter in range(iterations_per_model):
        modeldir = "{output_dir}/{extractor_type}/{feature_type}/{test_model}/noise-{noise_level}/iter-{iter}/".format(
            output_dir=output_dir,
            extractor_type=extractor_type,
            feature_type=feature_type,
            test_model=test_model,
            noise_level=noise_level,
            iter=iter)
        samples, cluster_indices, moved_atoms, feature_to_resids = \
            configuration.generate_data(test_model,
                                        noise_level,
                                        feature_type,
                                        nframes_per_cluster=1200  # TODO not necessary if files exist
                                        )
        feature_extractors = configuration.create_feature_extractors(extractor_type,
                                                                     samples=samples,
                                                                     cluster_indices=cluster_indices,
                                                                     n_splits=n_splits,
                                                                     n_iterations=n_iterations)
        all_postprocessors.append([])
        for i_extractor, extractor in enumerate(feature_extractors):
            do_computations = True
            if os.path.exists(modeldir):
                existing_files = glob.glob("{}/{}/accuracy*.npy".format(modeldir, extractor.name))
                if len(existing_files) > 0 and not overwrite:
                    logger.debug("File %s already exists. skipping computations", existing_files[0])
                    do_computations = False
            else:
                os.makedirs(modeldir)
            if do_computations:
                extractor.extract_features()
            pp = extractor.postprocessing(predefined_relevant_residues=moved_atoms,
                                          rescale_results=True,
                                          filter_results=False,
                                          working_dir=modeldir,
                                          feature_to_resids=feature_to_resids)
            if do_computations:
                pp.average()
                pp.evaluate_performance()
                pp.persist()
            else:
                pp.load()
            all_postprocessors[-1].append(pp)

    return np.array(all_postprocessors)
