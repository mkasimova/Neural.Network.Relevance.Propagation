from __future__ import absolute_import, division, print_function

import logging
import sys

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
    for pp in postprocessors:
        plt.plot(pp.index_to_resid, pp.importance_per_residue, label=pp.extractor.name)
        plt.xlabel("Residue")
        plt.ylabel("Importance")
        plt.legend()
    plt.show()
