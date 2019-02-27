# script to test the classifiers on zip code data

import numpy as np
from sklearn.metrics import accuracy_score
# work with zip code data
data_train = np.loadtxt('zip.train')
data_test = np.loadtxt('zip.test')

X_train = data_train[:,1:]
y_train = data_train[:,0]
X_test = data_test[:,1:]
y_test = data_test[:,0]

# get subset of data with digits 3 and 5 for training data
indices_of_rows_train = np.where(np.isin(y_train,[3,5]))
X_train_subset = X_train[indices_of_rows_train]
y_train_subset = y_train[indices_of_rows_train]

# get subset of data with digits 3 and 5 for testing data
indices_of_rows_test = np.where(np.isin(y_test,[3,5]))
print(indices_of_rows_test)
X_test_subset = X_test[indices_of_rows_test]
y_test_subset = y_test[indices_of_rows_test]


from compute_estimates import compute_estimates
from QDA_classifier import QDA_classifier

# do QDA on training and test data
QDA_estimates_train = compute_estimates(X_train_subset,y_train_subset)
y_train_pred_QDA = QDA_classifier(X_train_subset,QDA_estimates_train)


QDA_estimates_test = compute_estimates(X_test_subset,y_test_subset)
y_test_pred_QDA = QDA_classifier(X_test_subset, QDA_estimates_test)

# low accuracy and errors because variance matrix has very small values, and so computed determinant is very small
print("The training accuracy with QDA is: {}".format(accuracy_score(y_train_subset,y_train_pred_QDA)))
print("The testing accuracy with QDA is: {}".format(accuracy_score(y_test_subset,y_test_pred_QDA)))

from LDA_classifier import LDA_classifier

# do LDA on training and test data
LDA_estimates_train,variance_estimate_train = compute_estimates(X_train_subset,y_train_subset,classifier="LDA")
y_train_pred_LDA = LDA_classifier(X_train_subset,LDA_estimates_train,variance_estimate_train)

LDA_estimates_test,variance_estimate_test = compute_estimates(X_test_subset,y_test_subset,classifier="LDA")
y_test_pred_LDA = LDA_classifier(X_test_subset,LDA_estimates_test,variance_estimate_test)

print("The training accuracy with LDA is: {}".format(accuracy_score(y_train_subset,y_train_pred_LDA)))
print("The testing accuracy with LDA is: {}".format(accuracy_score(y_test_subset,y_test_pred_LDA)))

