from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import numpy as np

logger = logging.getLogger("utils-benchmarking")


def _to_numerical(postprocessors, postprocessor_to_number_func):
    res = np.empty(postprocessors.shape, dtype=float)
    for indices, pp in np.ndenumerate(postprocessors):
        num = postprocessor_to_number_func(pp)
        res[indices] = np.nan if num is None else num

    return res


def to_accuracy(postprocessors):
    return _to_numerical(postprocessors, lambda p: p.accuracy)


def to_accuracy_per_cluster(postprocessors):
    return _to_numerical(postprocessors, lambda p: p.accuracy_per_cluster)


def to_separation_score(postprocessors):
    return _to_numerical(postprocessors, lambda p: p.separation_score)


def find_best(postprocessors):
    accuracy = to_accuracy(postprocessors).mean(axis=0)
    ind = accuracy.argmax()
    return postprocessors[:, ind]


def strip_name(name):
    if name is None:
        return None
    parts = [
        n.split("-")[0]
        for n in name.split("_")
    ]
    return "\n".join(parts)
