import argparse
from modules import random_forest_features
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(epilog='Random forest classifier for feature importance extraction.')
parser.add_argument('-fe','--file_end_name',help='End file label.',default='')
parser.add_argument('-od','--out_directory',help='Folder where files are written.',default='')
parser.add_argument('-y','--cluster_indices',help='Cluster indices.',default='')
parser.add_argument('-f','--feature_list',help='Matrix with features [nSamples x nFeatures]',default='')
parser.add_argument('-n_iter','--number_of_iterations',help='Number of trainings to average the result over.',type=int,default=10)

args = parser.parse_args()

samples = np.load(args.feature_list)
cluster_indices = np.loadtxt(args.cluster_indices)

RF = random_forest_features.RF_feature_extract(samples, cluster_indices, n_iterations=args.number_of_iterations)

feature_importances, summed_FI = RF.extract_features()

ave_feature_importance = np.mean(summed_FI,axis=0)
std_feature_importance = np.std(summed_FI,axis=0)

np.savetxt(args.out_directory+'RF_features_'+args.file_end_name+'.txt',ave_feature_importance)

plt.figure(1)
plt.plot(ave_feature_importance)
plt.plot(ave_feature_importance+std_feature_importance)
plt.plot(ave_feature_importance-std_feature_importance)
plt.show()
