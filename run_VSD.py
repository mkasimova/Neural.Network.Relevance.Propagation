from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import os
import numpy as np

from modules import feature_extraction as fe, postprocessing as pp, visualization

logger = logging.getLogger("VSD")

working_dir = os.path.expanduser("~/projects/marina_demystifying/to_share.VSD") + '/'
data = np.load(working_dir + 'frame_i_j_contacts_dt1.npy')
cluster_indices = np.loadtxt(working_dir + 'clusters_indices.dat')

kwargs = {
    'samples': data,
    'cluster_indices': cluster_indices,
    'filter_by_distance_cutoff': True,
    'use_inverse_distances': True,
    'n_splits': 5,
    'n_iterations': 3,
    'scaling': True,
    'shuffle_datasets': True
}

feature_extractors = [
    fe.RandomForestFeatureExtractor(
        classifier_kwargs={
            'n_estimators': 100},
        **kwargs),
    fe.KLFeatureExtractor(bin_width=0.1, **kwargs),
    fe.MlpFeatureExtractor(
        classifier_kwargs={
            'hidden_layer_sizes': [100, ],
            'max_iter': 100000,
            'alpha': 0.0001},
        activation="relu",
        **kwargs)
]

common_peaks = {
    "R1-R4": [294, 297, 300, 303],
    "K5": [306],
    "R6": [309],
}
postprocessors = []
do_computations = True
filetype = "svg"
for extractor in feature_extractors:
    logger.info("Computing relevance for extractors %s", extractor.name)
    extractor.extract_features()
    p = pp.PostProcessor(extractor, working_dir=working_dir, pdb_file=working_dir + "alpha.pdb", filter_results=False)
    if do_computations:
        p.average()
        p.evaluate_performance()
        p.persist()
    else:
        p.load()

    visualization.visualize([[p]],
                            show_importance=True,
                            show_performance=False,
                            show_projected_data=False,
                            highlighted_residues=common_peaks,
                            outfile=working_dir + "{extractor}/importance_per_residue_{suffix}.{filetype}".format(
                                extractor=extractor.name,
                                filetype=filetype))
    if do_computations:
        visualization.visualize([[p]],
                                show_importance=False,
                                show_performance=True,
                                show_projected_data=False,
                                outfile=working_dir + "{extractor}/performance_{suffix}.{filetype}".format(
                                    extractor=extractor.name,
                                    filetype=filetype))

        visualization.visualize([[p]],
                                show_importance=False,
                                show_performance=False,
                                show_projected_data=True,
                                outfile=working_dir + "{extractor}/projected_data_{suffix}.{filetype}".format(
                                    extractor=extractor.name,
                                    filetype=filetype))

logger.info("Done")
