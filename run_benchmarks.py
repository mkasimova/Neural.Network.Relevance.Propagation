from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
from benchmarking import configuration

logger = logging.getLogger("benchmarking")


def run(extractor_type="KL",
        n_splits=1,
        n_iterations=5,
        feature_type='cartesian_rot',
        n_iter_per_example=10,
        test_model='linear',
        output_dir="output/benchmarking/"):
    test_noise = [1e-2, 1e-2, 2e-1, 2e-1]
    for j, j_noise in enumerate(test_noise):
        for k in range(n_iter_per_example):
            modeldir = "{}/{}/{}/{}/noise_{}/".format(output_dir, extractor_type, feature_type, test_model, j_noise)
            if not os.path.exists(modeldir):
                os.makedirs(modeldir)
            data, cluster_indices, moved_atoms, feature_to_resids = configuration.generate_data(test_model, j, j_noise,
                                                                                                feature_type)

            feature_extractors = configuration.create_feature_exractors(extractor_type,
                                                                        data=data,
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


if __name__ == "__main__":
    run()
