from __future__ import absolute_import, division, print_function

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

import os
import matplotlib.pyplot as plt
from . import utils

plt.style.use("seaborn-colorblind")

logger = logging.getLogger("viz_benchmarking")
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 18
_boxprops = dict(facecolor=[50.0 / 256.0, 117.0 / 256.0, 220.0 / 256.0])


def show_all(postprocessors, extractor_type,
             filename="all.svg",
             output_dir="output/benchmarking/"):
    if len(postprocessors) == 0:
        return
    output_dir = "{}/{}/".format(output_dir, extractor_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    supervised = postprocessors[0, 0].extractor.supervised
    xlabels = [utils.strip_name(pp.extractor.name) for pp in postprocessors[0]]
    fig, axs = plt.subplots(3 if supervised else 1, 1, sharex=True, sharey=True, squeeze=False)
    accuracy = utils.to_accuracy(postprocessors)
    axs[0, 0].boxplot(accuracy,
                      showmeans=True,
                      labels=xlabels,
                      patch_artist=True,
                      boxprops=_boxprops)
    axs[0, 0].set_ylabel("Accuracy")
    if supervised:
        axs[1, 0].boxplot(utils.to_accuracy_per_cluster(postprocessors),
                          showmeans=True,
                          labels=xlabels,
                          patch_artist=True,
                          boxprops=_boxprops)
        axs[1, 0].set_ylabel("Accuracy")
        axs[2, 0].boxplot(utils.to_separation_score(postprocessors),
                          showmeans=True,
                          labels=xlabels,
                          patch_artist=True,
                          boxprops=_boxprops)
        axs[2, 0].set_ylabel("Separation score")

    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir + filename)
    plt.clf()


def show_best(postprocessors, extractor_types,
              filename="all.svg",
              output_dir="output/benchmarking/"
              ):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    xlabels = extractor_types
    plt.boxplot(utils.to_accuracy(postprocessors),
                showmeans=True,
                labels=xlabels,
                patch_artist=True,
                boxprops=_boxprops)
    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir + filename)
    plt.clf()
