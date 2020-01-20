# Logistic Regression
This repository loads train and test samples for digit 7 and digit 8 from mnist dataset. It then performs classification on the data using Logistic Regression.

Once you clone the repository, download the dataset from https://drive.google.com/open?id=1L1SVmXSy_nnsdlxq7B2jRj9tPo387gke and put it inside the data folder.

An important step in logistic regression is calculating Log-Likelihood. Gradient ascent is used here for logistic regression. Gradient ascent is same as gradient descent with the only difference that a function is maximized instead of minimizing it.

During training phase, weights are calculated. Later, weights calculated in training phase are used for prediction during testing phase. Used parameters for logistic regression:
1. Learning rate = 0.001
2. Number of iterations: 400
