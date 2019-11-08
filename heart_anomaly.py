# Name: Jordan Le
# Date: 11-7-19
# Description: Project working with Naive Bayes on Binary Heart data.
#   - 0 is abnormal heart
#   - 1 is normal heart


# For Grading Run the following:
#   "make __________"
#   "make __________"
#   "make __________"


# Run with "python3 heart_anomalies.py"
# Installs "pip3 install <package>"


# Resources:
# Count 1's - https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero
# Understanding Naive Bayes - https://www.geeksforgeeks.org/naive-bayes-classifiers/


import numpy as np
import sys


# Read the text file data into 2d array. Give back 2d array and num.
# spect-resplit.train.csv
# spect-orig.train.csv
def read_data(data_file="heart-anomaly-hw/spect-resplit.train.csv"):
    ''' Read the csv file data into a 2d numpy array.
        Give back 2d array and the number of people.
        ndarray data
        int num_p
    '''
    # Numpy read in my data - separate by comma, all ints.  
    data = np.loadtxt(data_file, delimiter=",", dtype=int)
    num_p = len(data)
    print(data)
    print(num_p)

    labels = data[:,0]
    print(data[:,0])        # Display all the labels
    print(len(data[:,0]))   # Verify same number of samples

    print(np.count_nonzero(data[:,0]))  # count the number of 1's
    print("------------------------------------------")

    return data, num_p


def class_prob(labels):
    '''
        Get the probability of the heart being normal or abnormal.
    '''
    total_hearts = len(labels)                      # Total hearts
    normal_hearts = np.count_nonzero(labels)        # Count number of normal hearts
    abnormal_hearts = total_hearts - normal_hearts  # Abnormal hearts

    prob_normal = normal_hearts / float(total_hearts)      # Probability of normal hearts
    prob_abnormal = abnormal_hearts / float(total_hearts)  # Probability of abnormal hearts

    print ("""
    total_hearts = {}
    normal_hearts = {}
    abnormal_hearts = {}
    probs normal = {}%
    probs abnormal = {}%""".format(total_hearts, normal_hearts, abnormal_hearts, prob_normal, prob_abnormal))

    return normal_hearts, abnormal_hearts, prob_normal, prob_abnormal


def conditional_prob(labels, features, total_nrm, total_abn):
    '''
        Get the probability of normal or abnormal given a specific feature value.
        number of abnorm(0) & 0
        number of abnorm(0) & 1
        number of norm(1) & 0
        number of norm(1) & 1
        Labels compared to features. 
    '''
    #print(np.where(labels==1 and features==1))
    abn_0 = 0
    abn_1 = 0
    nrm_0 = 0
    nrm_1 = 0

    print(features)
    for i in range(len(labels)):
        # abnormal and 1
        if labels[i] == 0 and features[i] == 1:
            abn_1 += 1

        # normal and 1
        if labels[i] == 1 and features[i] == 1:
            nrm_1 += 1

    abn_0 = total_abn - abn_1
    nrm_0 = total_nrm - nrm_1

    prob_abn_0 = abn_0/float(total_abn) 
    prob_abn_1 = abn_1/float(total_abn) 
    prob_nrm_0 = nrm_0/float(total_nrm) 
    prob_nrm_1 = nrm_1/float(total_nrm) 

    print ("""
    abn_0 = {}      prob_abn_0 = {}
    abn_1 = {}      prob_abn_1 = {}
    nrm_0 = {}      prob_nrm_0 = {}
    nrm_1 = {}      prob_nrm_1 = {}
    """.format(abn_0, prob_abn_0, abn_1, prob_abn_1, nrm_0, prob_nrm_0, nrm_1, prob_nrm_1))
    pass





# Call Main
if __name__== "__main__" :
    data, num_p = read_data()
    
    normal_hearts, abnormal_hearts, prob_normal, prob_abnormal = class_prob(data[:,0])

    conditional_prob(data[:,0], data[:,1], normal_hearts, abnormal_hearts)
