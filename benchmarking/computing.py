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
from modules import visualization
from modules.data_generation import DataGenerator
from . import configuration

logger = logging.getLogger("ex_benchmarking")


def compute(extractor_type,
            n_splits=1,
            n_iterations=10,
            feature_type='cartesian_rot',
            iterations_per_model=10,
            test_model='linear',
            overwrite=False,
            accuracy_method='mse',
            displacement=1e-1,
            visualize=True,
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
    :param displacement: for toy model important atoms
    :param noise_level: for toy model frame generation
    :param output_dir:
    :return: postprocessors (np.array of dim iterations_per_model, nfeature_extractors)
    """
    all_postprocessors = []
    extractor_names = configuration.get_feature_extractors_names(extractor_type, n_splits=n_splits,
                                                                 n_iterations=n_iterations)
    n_extractors = len(extractor_names)
    for iter in range(iterations_per_model):
        modeldir = "{output_dir}/{extractor_type}/{feature_type}/{test_model}/noise-{noise_level}/iter-{iter}/".format(
            output_dir=output_dir,
            extractor_type=extractor_type,
            feature_type=feature_type,
            test_model=test_model,
            noise_level=noise_level,
            iter=iter)

        finished_extractors = []
        for name in extractor_names:
            if os.path.exists(modeldir):
                existing_files = glob.glob("{}/{}/importance_per_residue.npy".format(modeldir, name))
                if len(existing_files) > 0 and not overwrite:
                    logger.debug("File %s already exists. skipping computations", existing_files[0])
                    finished_extractors.append(name)
            else:
                os.makedirs(modeldir)
        needs_computations = len(finished_extractors) < n_extractors
        dg = DataGenerator(natoms=100,
                           nclusters=3,
                           natoms_per_cluster=[10, 10, 10],
                           nframes_per_cluster=1200 if needs_computations else 2,
                           # Faster generation for postprocessing purposes when we don't need the frames
                           test_model=test_model,
                           noise_natoms=None,
                           noise_level=noise_level,
                           displacement=displacement,
                           feature_type=feature_type)
        samples, labels = dg.generate_data()
        cluster_indices = labels.argmax(axis=1)

        feature_extractors = configuration.create_feature_extractors(extractor_type,
                                                                     samples=samples,
                                                                     cluster_indices=cluster_indices,
                                                                     n_splits=n_splits,
                                                                     n_iterations=n_iterations)
        all_postprocessors.append([])
        for i_extractor, extractor in enumerate(feature_extractors):
            do_computations = extractor.name not in finished_extractors
            if do_computations:
                extractor.extract_features()
            pp = extractor.postprocessing(predefined_relevant_residues=dg.moved_atoms,
                                          rescale_results=True,
                                          filter_results=False,
                                          working_dir=modeldir,
                                          accuracy_method=accuracy_method,
                                          feature_to_resids=dg.feature_to_resids())
            if do_computations:
                pp.average()
                pp.evaluate_performance()
                pp.persist()
            else:
                pp.load()
                pp.compute_accuracy()  # Recompute performance to handle changes in the accuracy measure
            if visualize:
                visualization.visualize([[pp]],
                                        show_importance=True,
                                        show_performance=False,
                                        show_projected_data=False,
                                        outfile="{}/{}/importance_per_residue.svg".format(modeldir, extractor.name),
                                        highlighted_residues=pp.predefined_relevant_residues.flatten(),
                                        show_average=False
                                        )
            all_postprocessors[-1].append(pp)

    return np.array(all_postprocessors)
