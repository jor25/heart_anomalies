# Name: Jordan Le
# Date: 11-7-19
# Description: Project working with Naive Bayes on Binary Heart data.
#   - 0 is abnormal heart
#   - 1 is normal heart


# For Grading Run the following:
#   "make run_all"


# Run with "python3 heart_anomalies.py <keyword>"
# keywords:
#   - SPECT                 - spect-itg
#   - spect-resplit-itg     - spect-orig
#   - spect-resplit 

# Installs "pip3 install <package>"


# Resources:
# Count 1's - https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero
# Understanding Naive Bayes - https://www.geeksforgeeks.org/naive-bayes-classifiers/
# Count matches in 2 arrays - https://stackoverflow.com/questions/42916330/efficiently-count-zero-elements-in-numpy-array


import numpy as np
import sys


def read_data(data_file="heart-anomaly-hw/spect-resplit.train.csv"):
    ''' Read the csv file data into a 2d numpy array.
        Give back 2d array and the number of people.
        ndarray data
        int num_p
    '''
    # Numpy read in my data - separate by comma, all ints.  
    data = np.loadtxt(data_file, delimiter=",", dtype=int)
    num_p = len(data)
    
    return data, num_p


def class_prob(labels):
    ''' Given labels, get the probability of the heart being normal or abnormal.
        Return number of normal hearts, abnormal hearts, and probabilities of both.
    '''
    total_hearts = len(labels)                              # Total hearts
    normal_hearts = np.count_nonzero(labels)                # Count number of normal hearts
    abnormal_hearts = total_hearts - normal_hearts          # Abnormal hearts

    prob_normal = normal_hearts / float(total_hearts)       # Probability of normal hearts
    prob_abnormal = abnormal_hearts / float(total_hearts)   # Probability of abnormal hearts

    return normal_hearts, abnormal_hearts, prob_normal, prob_abnormal


def conditional_prob(labels, features, total_nrm, total_abn):
    ''' Given labels, a feature column, the total number of normal hearts and 
        total number of abnormal hearts; get the probability of normal or abnormal 
        for the specific feature value.
        number of abnorm(0) & 0
        number of abnorm(0) & 1
        number of norm(1) & 0
        number of norm(1) & 1
        Labels compared to features. Also add .5 to each of the normal and 
        abnormal instances to prevent getting a log of zero.
        return 4 logged probabilities of the combinations above.
    '''
    # Initialize the different combinations
    abn_1 = len(np.where((labels==0) & (features==1))[0])   # Where abnormal and feature is one
    nrm_1 = len(np.where((labels==1) & (features==1))[0])   # Where normal and feature is zero
    abn_0 = total_abn - abn_1 + .5                          # Where abnormal if feature is zero + .5
    nrm_0 = total_nrm - nrm_1 + .5                          # Where normal if feature is one + .5
    abn_1 += .5                                             # Add .5 to abnormal to prevent zero denom
    nrm_1 += .5                                             # Add .5 to normal to prevent zero denom
    
    # Calculate the probabilities of each case
    prob_abn_0 = abn_0/float(total_abn) 
    prob_abn_1 = abn_1/float(total_abn) 
    prob_nrm_0 = nrm_0/float(total_nrm) 
    prob_nrm_1 = nrm_1/float(total_nrm) 
    
    # Give back two sets of probabilities, the series of normal or abnormal
    probs_nrms_01 = np.zeros(2)     # Normal probabilities
    probs_nrms_01[0] = prob_abn_0
    probs_nrms_01[1] = prob_abn_1

    probs_abrms_01 = np.zeros(2)    # Abnormal probabilities
    probs_abrms_01[0] = prob_nrm_0
    probs_abrms_01[1] = prob_nrm_1

    # Get the log of each of the probabilities in both arrays
    logged_probs = np.log2([probs_nrms_01, probs_abrms_01])

    # Return the log base 2 of each probability element
    return logged_probs



def classifier(test_data, probs_nora_01):
    '''
        Given the test data and the probabilities list of normal or abnormal, 0 or 1,
        predict classification of normal or abnormal heart. Return a list of predictions.
    '''
    predictions = []    # Initialize Predictions

    # Loop through all test instances. Note - watch out for off by 1 error
    for i in range(len(test_data)):                 # Loop through test data

        # Initialize and reset probabilities for each instance
        log_prob_nrm = 0        # Log probability of normal heart
        log_prob_abnrm = 0      # Log probability of abnormal heart

        # Loop through feature data (j-1 because features 22, test data 23)
        for j in range(1, (len(test_data[0]))):     
            log_prob_abnrm += probs_nora_01[j-1][0][test_data[i][j]]      # Probability of abnormal=0 heart
            log_prob_nrm += probs_nora_01[j-1][1][test_data[i][j]]        # Probability of normal=1 heart

        # Check which probability is greater and append predictions
        if log_prob_nrm > log_prob_abnrm:
            predictions.append(1)    # Heart Normal
        else:
            predictions.append(0)    # Heart Abnormal
        
    # Return the list of predictions for the test data
    return predictions


def true_pos_neg(predictions, labels, nora):
    ''' Given a list of predictions, labels, and a normal or abnormal int value;
        Return the number of correct, the total, and the precentage of them in a list.
    '''
    # Conditions:
    cond_1 = (np.asarray(predictions) == nora)
    cond_2 = (labels == nora)

    # Lists of specific matches
    matches = np.where(cond_1 & cond_2)
    num_abnorm_test = np.count_nonzero(labels == nora)  # Count number of abnormal hearts in test data
    
    # Number of abnormal correct, number of total abnormal, and percentage in a list
    return [ len(matches[0]), num_abnorm_test, len(matches[0])/float(num_abnorm_test) ]


def parser(user_cmd):
    ''' Given the user commands, parse them to determine the proper filenames.
        Then return the training file, testing file, and the user commands.
    '''
    # Parse the user input
    if user_cmd[1] == "SPECT":
        train_file = "heart-anomaly-hw/" + user_cmd[1] + ".train"
        test_file = "heart-anomaly-hw/" + user_cmd[1] + ".test"
    else:
        train_file = "heart-anomaly-hw/" + user_cmd[1] + ".train.csv"
        test_file = "heart-anomaly-hw/" + user_cmd[1] + ".test.csv"

    # Return training file, test file, and user commands
    return train_file, test_file, user_cmd


def disp_write_out(user_cmd, accuracy, true_neg, true_pos):
    ''' Given the user command, accuracy, true negative, and true positives,
        display the information to standard out and write the output to a file.
    '''
    output = "{} {}/{}({}) {}/{}({}) {}/{}({})".format(user_cmd[1], accuracy[0], accuracy[1], round(accuracy[2], 2),
                                                    true_neg[0], true_neg[1], round(true_neg[2], 2),
                                                    true_pos[0], true_pos[1], round(true_pos[2], 2))
    # Make a file and write out
    out_file = "final_scores/out_{}".format(user_cmd[1]) 
    fout = open(out_file, "w")      # Output file
    fout.write(output)              # Write txt_line to the file
    fout.close()                    # Close the file

    # Display to standard output
    print(output)


# Call Main and do the good stuff
if __name__== "__main__" :

    # Parse user input and provide file names. Don't use SPECTF - not binary 
    train_file, test_file, user_cmd = parser(sys.argv)
    
    # Initialize list of probability of normal or abnormal with 0 or 1
    probs_nora_01 = []      

    # Read in the data from the training file
    data, num_p = read_data(train_file)
    
    # Prework for determining probabilities from training data.
    normal_hearts, abnormal_hearts, prob_normal, prob_abnormal = class_prob(data[:,0])
    
    # Loop through all the other features and determine a probability for each
    for i in range(1, (len(data[0]))):
        probs_nora_01.append( conditional_prob(data[:,0], data[:,i], normal_hearts, abnormal_hearts) )
    
    # Now get my test data
    test_data, num_p2 = read_data(test_file)
   
    # Classify data instances and get prediction
    predictions = classifier(test_data, probs_nora_01)

    # The number of correct for accuracy.
    verify_list = np.equal(test_data[:,0], predictions)

    # Merge accuracy
    accuracy = [np.sum(verify_list), len(test_data), np.sum(verify_list)/ float(len(test_data))]
    
    # Verify number of abnormal hearts identified correctly
    true_neg = true_pos_neg(predictions, test_data[:,0], 0)

    # Verify number of normal hearts identified correctly
    true_pos = true_pos_neg(predictions, test_data[:,0], 1)

    # Display to standard out and write to output file
    disp_write_out(user_cmd, accuracy, true_neg, true_pos)
