from __future__ import absolute_import, division, print_function

import sys
import logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s %(name)s-%(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
import numpy as np
from scipy.spatial.distance import squareform
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import modules.relevance_propagation

logger = logging.getLogger("utils")

'''

CODE BODY

'''



def vectorize(data):
    """
    Vectorizes the input
    """
    if (len(data.shape))==3 and (data.shape[1]==data.shape[2]):
        data_vect = []
        for i in range(data.shape[0]):
            data_vect.append(squareform(data[i,:,:]))
        data_vect = np.asarray(data_vect)
    elif (len(data.shape))==2:
        data_vect = data
    else:
        sys.exit("The input array has wrong dimensionality")
    return data_vect


def convert_to_contact(data,cutoff=0.5):
    """
    Converts distances to contacts with a chosen cutoff
    """

    data = 1/data

    logger.info("Converting distances to contacts with a %s nm cutoff ...", cutoff)
    if len(data.shape)==2:
        data_cont = np.zeros((data.shape[0],data.shape[1]))
        for i in range(data.shape[0]):
            ind_1 = np.where(data[i]<=cutoff)
            data_cont[i,ind_1] = 1

    elif len(data.shape)==3 and (data.shape[1]==data.shape[2]):
        data_cont = np.zeros((data.shape[0],data.shape[1],data.shape[2]))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ind_1 = np.where(data[i,j]<=cutoff)
                data_cont[i,j,ind_1] = 1

    else:
        sys.exit("The input array has wrong dimensionality")

    return data_cont


def scale(data,perc_2=None,perc_98=None,scaler=None):
    """
    Scales the input and removes outliers
    """

    if perc_2 is None and perc_98 is None:
        perc_2 = np.zeros(data.shape[1])
        perc_98 = np.zeros(data.shape[1])

        for i in range(data.shape[1]):
            perc_2[i] = np.percentile(data[:,i],2)
            perc_98[i] = np.percentile(data[:,i],98)

    for i in range(data.shape[1]):

        perc_2_ind = np.where(data[:,i]<perc_2[i])[0]
        perc_98_ind = np.where(data[:,i]>perc_98[i])[0]
        data[perc_2_ind,i] = perc_2[i]
        data[perc_98_ind,i] = perc_98[i]

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)

    data_scaled = scaler.transform(data)
    return data_scaled, perc_2, perc_98, scaler


def transform_to_matrix(clustering):
    """
    Transforms a vector of cluster indices to a matrix where a 1 on the ij element means that the ith frame was in cluster state j+1
    """
    number_of_clusters = len(set([t for t in clustering]))
    T = np.zeros((len(clustering), number_of_clusters), dtype=int)
    for i in range(0, len(clustering)):
        T[i, int(clustering[i] - 1)] = 1
    return T


def train_NN(data_scaled,
             clustering_prob,
             hidden_layer_sizes,
             randomize=False):
    """
    Trains a classifier
    If randomize is set to False the classifier will not use a constant random seed
    """
    classifier = MLPClassifier(
        solver='lbfgs',
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=(None if randomize else 89274),
        activation='relu',
        max_iter=500)
    classifier.fit(data_scaled,clustering_prob)
    return classifier


def check_for_overfit(data_scaled,clustering_prob,classifier):
    """
    Checks if the classifier is overfitted
    Computes an error in the form: sum(1-Ptrue+sum(Pfalse))/N_clusters
    """

    clustering_predicted = classifier.predict(data_scaled)

    # Calculate the error as sum(1-Ptrue+sum(Pfalse))/N_clusters
    error_per_frame = np.zeros((clustering_prob.shape[0]))
    number_of_clusters = clustering_prob.shape[1]

    for i in range(clustering_prob.shape[0]):
        error_per_frame[i] = (1-np.dot(clustering_prob[i],clustering_predicted[i])+\
                              np.dot(1-clustering_prob[i],clustering_predicted[i]))/\
                              number_of_clusters
    error = np.average(error_per_frame)*100
    return error


def perform_relevance_propagation(data_vect,\
                       clustering,\
                       hidden_layer_sizes,\
                       n_iter,\
                       n_halves,\
                       scaling=True):

    number_of_features = data_vect.shape[1]
    clustering_labels = np.unique(clustering)
    number_of_clusters = len(clustering_labels)
    relevance_av_cluster = np.zeros((n_iter*n_halves,number_of_clusters,number_of_features))
    error_array = np.zeros((n_iter*n_halves))
    ind_to_keep = []
    relevance_av_cluster_iter = np.zeros((number_of_clusters,number_of_features,2))

    logger.info("Performing relevance propagation for the dataset with %s features and %s samples", number_of_features, data_vect.shape[0])
    logger.info("Number of clusters is %s", number_of_clusters)

    # Do n_iter iteration, in each the neural network is trained on one half of the dataset and then
    # it is checked for overfitting using the other half
    # If both halves are used for relevance estimation the overall number of iterations is n_iter*n_halves
    # If the neural network is not overfitted, it is used to calculate the relevance of the collective variables

    for i in range(n_iter):

        logger.info("Running iteration %s ...", i+1)

        # Re-shuffle the input dataset and take one half of the data using stride 2 on the re-shuffled dataset
        X, Y = shuffle(data_vect,clustering,random_state=89274)
        data_first_half = X[0::2,:]
        data_second_half = X[1::2,:]
        clustering_first_half = Y[0::2]
        clustering_second_half = Y[1::2]

        # How many halves you want to use to estimate relevance - one or two?
        for j in range(n_halves):
            if j==0:
                data_train = np.copy(data_first_half)
                data_test = np.copy(data_second_half)
                clustering_train = np.copy(clustering_first_half)
                clustering_test = np.copy(clustering_second_half)
            elif j==1:
                data_train = np.copy(data_second_half)
                data_test = np.copy(data_first_half)
                clustering_train = np.copy(clustering_second_half)
                clustering_test = np.copy(clustering_first_half)

            if scaling==True:
                logger.info("			Scaling the input dataset ...")
                # Scale the input dataset using MinMaxScaler (change if needed) and remove outliers
                data_train_scaled, perc_2, perc_98, scaler = scale(data_train)
            else:
                data_train_scaled = data_train

            # Transform clusters into probabilities
            clustering_train_prob = transform_to_matrix(clustering_train.astype(int))

            logger.info("			Training the neural network ...")

            # Train NN
            classifier = train_NN(data_train_scaled,\
                                  clustering_train_prob,\
                                  hidden_layer_sizes,\
                                  randomize=True)

            logger.info("			Checking for overfit ...")

            # Check for overfitting
            if scaling==True:
                data_test_scaled, perc_2, perc_98, scaler = scale(data_test,\
                                                                  perc_2,\
                                                                  perc_98,\
                                                                  scaler)
            else:
                data_test_scaled = data_test

            clustering_test_prob = transform_to_matrix(clustering_test.astype(int))

            error = check_for_overfit(data_test_scaled,\
                                      clustering_test_prob,\
                                      classifier)

            error_array[i*n_halves+j] = error

            logger.info("			Overfit error is %s", error)

            # If error is too high, re-train the neural network, else - continue with relevance propagation
            if error<=5:

                logger.info("			Error is less than 5%, therefore computing relevance ...")

                ind_to_keep.append(i*n_halves+j)

                # Infer the weights and biases of the neural network
                weights = classifier.coefs_
                biases = classifier.intercepts_

                data_propagation = np.copy(data_train_scaled)
                clustering_propagation = np.copy(clustering_train_prob)

                # Calculate relevance
                relevance = Relevance_Propagation.relevance_propagation(weights,\
                                                                        biases,\
                                                                        data_propagation,
                                                                        clustering_propagation)

                logger.info("			Rescaling relevance according to min and max in each frame ...")
                logger.info("			... and averaging it over each cluster")

                # Rescale relevance according to min and max relevance in each frame
                for k in range(relevance.shape[0]):
                    relevance[k,:] = (relevance[k,:]-np.min(relevance[k,:]))/\
                                                        (np.max(relevance[k,:])-np.min(relevance[k,:])+0.000000001)

                # Average relevance over each cluster
                for k in range(number_of_clusters):
                    relevance_av_cluster[i*n_halves+j,k,:] = \
                                                    np.mean(relevance[np.where(clustering_train==clustering_labels[k])[0]],axis=0)

    logger.info("Done with iterations!")
    logger.info("Computing relevance average and std over all iterations ...")

    # Calculate average and error of relevance over all iterations
    # Calculate ave and err only for those CVs, for which at least n_iter/2 iterations have been converged
    if len(ind_to_keep)>=(n_iter*n_halves)/2:
        relevance_av_cluster = relevance_av_cluster[ind_to_keep,:,:]
        relevance_av_cluster_iter[:,:,0] = np.average(relevance_av_cluster,axis=0)
        relevance_av_cluster_iter[:,:,1] = np.std(relevance_av_cluster,axis=0)

    logger.info("Done with computing relevance!")

    return relevance_av_cluster_iter, error_array


def write_results_input_matrix(relevance,home_dir,fid,DKL=None,only_significant=False,sigma=2):
    '''
    Write results if input is a set of square matrices
    '''

    logger.info("Writing the results ...")
    logger.info("Sigma based filtering of relevances is %s",only_significant)

    number_of_clusters = relevance.shape[0]
    number_of_features = (squareform(relevance[0,:,0])).shape[0]

    relevance_ave_per_residue = np.zeros((number_of_features,number_of_clusters))
    relevance_std_per_residue = np.zeros((number_of_features,number_of_clusters))

    for i in range(number_of_clusters):
        if only_significant==True:
            ind_nonzero = np.where(relevance[i,:,0]>0)
            relevance_global_mean = np.mean(relevance[i,ind_nonzero,0])
            relevance_global_sigma = np.std(relevance[i,ind_nonzero,0])

        relevance_ave_matrix = squareform(relevance[i,:,0])
        relevance_std_matrix = squareform(relevance[i,:,1])

        for j in range(number_of_features):
            if only_significant==True:
                ind_above_sigma = np.where(relevance_ave_matrix[j,:]>=\
                                      (relevance_global_mean+sigma*relevance_global_sigma))[0]
                relevance_ave_per_residue[j,i] = np.sum(relevance_ave_matrix[j,ind_above_sigma])
                relevance_std_per_residue[j,i] = np.sqrt(np.sum(relevance_std_matrix[j,ind_above_sigma]**2))
            else:
                relevance_ave_per_residue[j,i] = np.sum(relevance_ave_matrix[j,:])
                relevance_std_per_residue[j,i] = np.sqrt(np.sum(relevance_std_matrix[j,:]**2))

    results = np.zeros((number_of_features,number_of_clusters*2))

    for i in np.arange(number_of_clusters):
        results[:,2*i] = relevance_ave_per_residue[:,i]
        results[:,2*i+1] = relevance_std_per_residue[:,i]

    np.savetxt(home_dir+fid,results)

    if DKL is not None:
        DKL_square = squareform(DKL)
        DKL_mean = np.mean(DKL)
        DKL_std = np.std(DKL)
        DKL_out = []
        for i in range(DKL_square.shape[0]):
            DKL_ind_above_sigma = np.where(DKL_square[i,:]>=DKL_mean+2*DKL_std)[0]
            DKL_out.append(np.sum(DKL_square[i,DKL_ind_above_sigma]))

        DKL_out = np.asarray(DKL_out).T

        np.savetxt(home_dir+'DKL.'+fid,DKL_out)

    logger.info("Done!")