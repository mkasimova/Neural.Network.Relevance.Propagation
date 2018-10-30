from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import matplotlib.pyplot as plt

logger = logging.getLogger("postprocessing")


def visualize(postprocessors):
    """
    Plots the feature per residue.
    TODO visualize features too with std etc
    :param postprocessors:
    :return:
    """

    n_feature_extractors = len(postprocessors)
    dx = np.arange(-0.3,0.3,n_feature_extractors)
    cols =np.asarray([[0,0,0],[0.5,0,0],[0,0.5,0],[0,0,0.5]])

    counter = 0
    for pp in postprocessors:
        #plt.plot(pp.index_to_resid, pp.importance_per_residue, label=pp.extractor.name)

        x_val = pp.index_to_resid+dx[counter]
        y_val = pp.importance_per_residue
        std_val = pp.std_importance_per_residue

        plt.bar(x_val, y_val,color=cols[counter,:], label=pp.extractor.name)
        plt.plot(x_val, y_val+std_val,color='k',marker='s',markersize=1)
        plt.plot(x_val, y_val-std_val, color='k', marker='s', markersize=1)
        plt.xlabel("Residue")
        plt.ylabel("Importance")
        counter+=1
    plt.legend()
    plt.show()
