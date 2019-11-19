# heart_anomalies
# Project Description:
    Implement a Naive Bayes Classifier for binary heart data.
    Classify if they are abnormal or normal hearts and compare the test data with that of the predictions.
    Output the results of all datasets with:

        "File_Id Accuracy% (Correct/Total) True Negative% (Correct Abnormal/Total Abnormal) True Positive% (Correct Normal/Total Normal)"

    Refer to the `heart-anomaly-hw/heart.pdf` for assignment description.

# How to run:
    `make run_all`
    - This goes through all of the files and determines their scoring.
    - All except SPECTF - because this one is not binary.

# References:
    Count the number of ones in a numpy array:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero

    Step by step refresher on understanding Naive Bayes probability calculations:
        https://www.geeksforgeeks.org/naive-bayes-classifiers/

    Count the number of matches in two numpy arrays:
        https://stackoverflow.com/questions/42916330/efficiently-count-zero-elements-in-numpy-array

    Basic logarithm rules:
        https://www.chilimath.com/lessons/advanced-algebra/logarithm-rules/

