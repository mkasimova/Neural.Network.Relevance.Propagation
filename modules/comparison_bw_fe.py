from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np
from scipy.optimize import curve_fit

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("comparison_bw_fe")

def func(x, a, b, c):
    return np.exp(-a*x)*b+c

def compare(postprocessors):

    # Set parameters
    # Number of top features to be compared per iteration; in %
    n_features_per_iteration_perc = 0.01
    n_features = len(postprocessors[0].feature_importance[:,0])-len(postprocessors[0].indices_filtered)
    n_features_per_iteration = int(n_features_per_iteration_perc*n_features)
    # Shift during rolling, in number of features
    n_features_shift = 1

    # Consider up to 75% of all features
    # TODO: Why does it start at n_features_per_iteration?
    top_features_array = np.arange(n_features_per_iteration,int(0.75*n_features),n_features_shift)

    n_features_extractors = len(postprocessors)

    difference_rank = np.zeros((len(top_features_array)-1,n_features_extractors,n_features_extractors))
    top_features_average = np.zeros((len(top_features_array)-1,n_features_extractors))

    for i in range(len(top_features_array)-1):

        for j in range(n_features_extractors):

            p1 = postprocessors[j]
            p1_feature_importance = np.mean(p1.feature_importance,axis=1)
            p1_ind_top_features = np.argsort(-p1_feature_importance)[top_features_array[i]:top_features_array[i]+n_features_per_iteration]
            p1_top_features = p1_feature_importance[p1_ind_top_features]

            top_features_average[i,j] = np.average(p1_top_features)

            for k in range(j+1, n_features_extractors):

                p2 = postprocessors[k]
                p2_feature_importance = np.mean(p2.feature_importance,axis=1)
                p2_ind_top_features = np.argsort(-p2_feature_importance)[top_features_array[i]:top_features_array[i]+n_features_per_iteration]
                p2_top_features = p2_feature_importance[p2_ind_top_features]

                # Compare how much p1_ind_top_features and p2_ind_top_features overlap
                overlap_ind = np.intersect1d(p1_ind_top_features,p2_ind_top_features)

                # TODO: Or fraction of overlap?
                difference_rank[i,j,k] = (len(p1_top_features)+len(p2_top_features)-2*len(overlap_ind))/(len(p1_top_features)+len(p2_top_features))
                difference_rank[i,k,j] = difference_rank[i,j,k]

    # Average over all features extractors
    # TODO ?: difference_rank = np.mean(difference_rank,axis=(1,2))
    difference_rank = np.sum(np.sum(difference_rank,axis=2),axis=1)/(n_features_extractors-1)/n_features_extractors
    difference_rank = 1 - difference_rank # TODO: If we would you fraction, this would not be needed, I think.

    x = (top_features_array - n_features_per_iteration)[:-1]+n_features_per_iteration # TODO: Isn't this the same as top_features_array[:-1], that is all values except the last?
    diff_popt, diff_pcov = curve_fit(func, x, difference_rank)
    difference_rank_fit = func(x, *diff_popt)

    return x, difference_rank, difference_rank_fit
