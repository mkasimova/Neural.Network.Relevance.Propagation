from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

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
    # Shift during rolling, in n features
    n_features_shift = 1

    # Consider up to 75% of all features
    top_features_array = np.arange(n_features_per_iteration,int(0.75*n_features),n_features_shift)

    n_features_extractors = len(postprocessors)

    #difference = np.zeros((len(top_features_array)-1,n_features_extractors,n_features_extractors))
    difference_rank = np.zeros((len(top_features_array)-1,n_features_extractors,n_features_extractors))
    top_features_average = np.zeros((len(top_features_array)-1,n_features_extractors))

    for i in range(len(top_features_array)-1):

        for j in range(n_features_extractors):

            p1 = postprocessors[j]
            # averages features importances over all clusters ...
            p1_feature_importance = np.mean(p1.feature_importance,axis=1)
            p1_ind_top_features = np.argsort(-p1_feature_importance)[top_features_array[i]:top_features_array[i]+n_features_per_iteration]
            p1_top_features = p1_feature_importance[p1_ind_top_features]

            top_features_average[i,j] = np.average(p1_top_features)

            for k in range(j+1, n_features_extractors):

                p2 = postprocessors[k]
                # averages features importances over all clusters ...
                p2_feature_importance = np.mean(p2.feature_importance,axis=1)
                p2_ind_top_features = np.argsort(-p2_feature_importance)[top_features_array[i]:top_features_array[i]+n_features_per_iteration]
                p2_top_features = p2_feature_importance[p2_ind_top_features]

                # Compare how much p1_ind_top_features and p2_ind_top_features overlap
                overlap_ind = np.intersect1d(p1_ind_top_features,p2_ind_top_features)

                #difference[i,j,k] = (np.sum(p1_top_features+p2_top_features)-np.sum(p1_feature_importance[overlap_ind]+p2_feature_importance[overlap_ind]))/np.sum(p1_top_features+p2_top_features)
                #difference[i,k,j] = difference[i,j,k]

                difference_rank[i,j,k] = (len(p1_top_features)+len(p2_top_features)-2*len(overlap_ind))/(len(p1_top_features)+len(p2_top_features))
                difference_rank[i,k,j] = difference_rank[i,j,k]

    # Average over all features extractors
    difference_rank = np.sum(np.sum(difference_rank,axis=2),axis=1)/(n_features_extractors-1)/n_features_extractors
    difference_rank = 1 - difference_rank
    top_features_average = np.average(top_features_average,axis=1)

    x = (top_features_array - n_features_per_iteration)[:-1]

    diff_popt, diff_pcov = curve_fit(func, x, difference_rank)
    ave_popt, ave_pcov = curve_fit(func, x, top_features_average)

    logger.info("Relevance decay of difference between fe is %s, error is %s", diff_popt[0], np.sqrt(np.diag(diff_pcov))[0])
    diff_n_filtered_features = int(-1/diff_popt[0]*np.log(0.5))+n_features_per_iteration
    logger.info("Number of significant feature importances is %s", diff_n_filtered_features)
    logger.info("Relevance decay of average importance for all fe is %s, error is %s", ave_popt[0], np.sqrt(np.diag(ave_pcov))[0])
    ave_n_filtered_features = int(-1/ave_popt[0]*np.log(0.5))+n_features_per_iteration
    logger.info("Number of significant feature importances is %s", ave_n_filtered_features)

    plt.plot(x+n_features_per_iteration,difference_rank,label='1-difference')
    plt.plot(x+n_features_per_iteration,top_features_average,label='average importance')
    plt.plot(x+n_features_per_iteration,func(x, *diff_popt),label='1-diff FIT')
    plt.plot(x+n_features_per_iteration,func(x, *ave_popt),label='1-ave FIT')
    plt.xlabel('top features')
    plt.ylabel('1-diff or ave')
    plt.xscale('log')
    plt.legend()
    plt.show()

    return ave_n_filtered_features
