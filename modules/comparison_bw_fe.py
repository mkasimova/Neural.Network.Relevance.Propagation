from __future__ import absolute_import, division, print_function

import logging
import sys
import numpy as np

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger("comparison_bw_fe")

def compare(postprocessors):

    n_features_extractors = len(postprocessors)

    n_features = len(postprocessors[0].feature_importance[:,0])-len(postprocessors[0].indices_filtered)

    n_top_features = (n_features*np.arange(0.01,1,0.01)).astype(int)

    difference = np.zeros((len(n_top_features)-1,n_features_extractors,n_features_extractors))
    difference_rank = np.zeros((len(n_top_features)-1,n_features_extractors,n_features_extractors))

    for i in range(len(n_top_features)-1):

        for j in range(n_features_extractors):

            p1 = postprocessors[j]
            p1_feature_importance = np.mean(p1.feature_importance,axis=1) #TODO averages features importances over all clusters ...
            p1_ind_top_features = np.argsort(-p1_feature_importance)[n_top_features[i]:n_top_features[i+1]]
            p1_top_features = p1_feature_importance[p1_ind_top_features]

            for k in range(j+1, n_features_extractors):

                p2 = postprocessors[k]
                p2_feature_importance = np.mean(p2.feature_importance,axis=1) #TODO averages features importances over all clusters ...
                p2_ind_top_features = np.argsort(-p2_feature_importance)[n_top_features[i]:n_top_features[i+1]]
                p2_top_features = p2_feature_importance[p2_ind_top_features]

                # Compare how much p1_ind_top_features and p2_ind_top_features overlap

                overlap_ind = np.intersect1d(p1_ind_top_features,p2_ind_top_features)

                difference[i,j,k] = (np.sum(p1_top_features+p2_top_features)-np.sum(p1_feature_importance[overlap_ind]+p2_feature_importance[overlap_ind]))/np.sum(p1_top_features+p2_top_features)
                difference[i,k,j] = difference[i,j,k]

                difference_rank[i,j,k] = (len(p1_top_features)+len(p2_top_features)-2*len(overlap_ind))/(len(p1_top_features)+len(p2_top_features))
                difference_rank[i,k,j] = difference_rank[i,j,k]

    directory = postprocessors[0].working_dir + "analysis/"
    np.save(directory+'difference.npy',difference)
    np.save(directory+'difference_rank.npy',difference_rank)
