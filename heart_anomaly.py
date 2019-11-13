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
# Count matches in 2 arrays - https://stackoverflow.com/questions/42916330/efficiently-count-zero-elements-in-numpy-array


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
    #print(data)
    #print(num_p)

    labels = data[:,0]
    '''
    print(data[:,0])        # Display all the labels
    print(len(data[:,0]))   # Verify same number of samples

    print(np.count_nonzero(data[:,0]))  # count the number of 1's
    print("------------------------------------------")
    '''
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
    abn_1 = len(np.where((labels==0) & (features==1))[0])
    nrm_1 = len(np.where((labels==1) & (features==1))[0])
    abn_0 = total_abn - abn_1 + .5
    nrm_0 = total_nrm - nrm_1 + .5
    abn_1 += .5                                             # Add .5 to abnormal to prevent zero denom
    nrm_1 += .5                                             # Add .5 to normal to prevent zero denom
    
    #print(features)

    # Calculate the probabilities of each case
    prob_abn_0 = abn_0/float(total_abn) 
    prob_abn_1 = abn_1/float(total_abn) 
    prob_nrm_0 = nrm_0/float(total_nrm) 
    prob_nrm_1 = nrm_1/float(total_nrm) 
    '''
    print ("""
    abn_0 = {}      prob_abn_0 = {}
    abn_1 = {}      prob_abn_1 = {}
    nrm_0 = {}      prob_nrm_0 = {}
    nrm_1 = {}      prob_nrm_1 = {}
    """.format(abn_0, prob_abn_0, abn_1, prob_abn_1, nrm_0, prob_nrm_0, nrm_1, prob_nrm_1))
    '''
    # Give back two sets of probabilities, the series of normal or abnormal
    probs_nrms_01 = np.zeros(2)     # Normal probabilities
    probs_nrms_01[0] = prob_abn_0
    probs_nrms_01[1] = prob_abn_1

    probs_abrms_01 = np.zeros(2)    # Abnormal probabilities
    probs_abrms_01[0] = prob_nrm_0
    probs_abrms_01[1] = prob_nrm_1

    #print(probs_nrms_01, probs_abrms_01)
    # Get the log of each of the probabilities in both arrays
    logged_probs = np.log2([probs_nrms_01, probs_abrms_01])
    #print(logged_probs)

    # Return the log base 2 of each probability element
    return logged_probs



def classifier(test_data, probs_nora_01):
    '''
        Given the test data and the probabilities list of normal or abnormal, 0 or 1,
        predict classification of normal or abnormal heart.
    '''
    log_prob_nrm = 0
    log_prob_abnrm = 0
    predictions = []

    # Note - watch out for off by 1 error
    for i in range(len(test_data)):                 # Loop through test data
        for j in range(1, (len(test_data[0]))):     # Loop through feature data (j-1 because features 22, test data 23)
            #print("i = {}\tj = {}\tlen test[0] = {}\tlen probs_nora = {}\ttest[i][j] = {}".format(i, j, len(test_data[0]), len(probs_nora_01), test_data[i][j]))
            log_prob_abnrm += probs_nora_01[j-1][0][test_data[i][j]]      # Probability of abnormal=0 heart
            log_prob_nrm += probs_nora_01[j-1][1][test_data[i][j]]        # Probability of normal=1 heart

        # Check which probability is greater
        if log_prob_nrm > log_prob_abnrm:
            predictions.append(1)    # Heart Normal
        else:
            predictions.append(0)    # Heart Abnormal
        
        # Reset probabilities for each
        log_prob_nrm = 0
        log_prob_abnrm = 0

    # Return the list of predictions for the test data
    return predictions


def true_pos_neg(predictions, labels, nora):
    ''' 
        Given a list of predictions, labels, and a normal or abnormal value;
        Return the number of correct, the total, and the precentage of them.
    '''
    # Conditions:
    cond_1 = (np.asarray(predictions) == nora)
    cond_2 = (labels == nora)

    part_2 = np.where(cond_1 & cond_2)
    #print(part_2)
    num_abnorm_test = np.count_nonzero(labels == nora)        # Count number of abnormal hearts in test data
    #print("Abnormal/Total Abnormal: {}/{}".format(len(part_2[0]), num_abnorm_test))
    
    # Number of abnormal correct, number of total abnormal, percentage
    return [ len(part_2[0]), num_abnorm_test, len(part_2[0])/float(num_abnorm_test) ]



# Call Main
if __name__== "__main__" :
    # don't use SPECTF - not binary 
    user_cmd = sys.argv
    print(user_cmd)
    if user_cmd[1] == "SPECT":
        train_file = "heart-anomaly-hw/" + user_cmd[1] + ".train"
        test_file = "heart-anomaly-hw/" + user_cmd[1] + ".test"
    else:
        train_file = "heart-anomaly-hw/" + user_cmd[1] + ".train.csv"
        test_file = "heart-anomaly-hw/" + user_cmd[1] + ".test.csv"
        
    print ("train: {}\ntest: {}".format(train_file, test_file))
    
    probs_nora_01 = []      # Probability of normal or abnormal with 0 or 1

    #data, num_p = read_data("heart-anomaly-hw/spect-orig.train.csv")
    #data, num_p = read_data("heart-anomaly-hw/SPECT.train")
    data, num_p = read_data(train_file)
    
    # Prework for determining probabilities from training data.
    normal_hearts, abnormal_hearts, prob_normal, prob_abnormal = class_prob(data[:,0])
    
    # Loop through all the other features and determine a probability for each
    for i in range(1, (len(data[0]))):
        #print("{}/{}".format(i, len(data[0])-1 ))   # Show what feature I'm on with this count from 1.
        probs_nora_01.append( conditional_prob(data[:,0], data[:,i], normal_hearts, abnormal_hearts) )
    
    '''
    # Verify data for classifier
    print (probs_nora_01)
    print (len(probs_nora_01))      # 22 features
    print (probs_nora_01[21][0][0]) # how to access feature 21, abnormal=0 or normal=1 (do both), if it's 0 or 1 on feature. 
    '''

    # Now get my test data
    # heart-anomaly-hw/spect-orig.test.csv
    #test_data, num_p2 = read_data("heart-anomaly-hw/spect-resplit.test.csv")
    #test_data, num_p2 = read_data("heart-anomaly-hw/spect-orig.test.csv")
    #test_data, num_p2 = read_data("heart-anomaly-hw/SPECT.test")
    test_data, num_p2 = read_data(test_file)
   
    # Classify data instances and get prediction
    predictions = classifier(test_data, probs_nora_01)

    # The number of correct for accuracy.
    verify_list = np.equal(test_data[:,0], predictions)

    # Display accuracy
    print("Correct/Total: {}/{} ({})".format(np.sum(verify_list), len(test_data), np.sum(verify_list)/ float(len(test_data))))
    
    # Verify number of abnormal hearts identified correctly
    print("True Negative: ", true_pos_neg(predictions, test_data[:,0], 0))

    # Verify number of normal hearts identified correctly
    print("True Positive: ", true_pos_neg(predictions, test_data[:,0], 1))
