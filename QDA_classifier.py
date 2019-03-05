# function to compute QDA classifier
# coded in Python

import numpy as np
# allows us to multiply more than one vector at once
from numpy.linalg import multi_dot
# allows us to compute inverse of matrix
from numpy.linalg import inv
# allows us to compute determinant of matrix
from numpy.linalg import det

def QDA_classifier(X,estimates):
	"""
	A function to return LDA classification output for a given X
	We won't use a vectorized implementation here because it complicates
	things when dealing with the dimensions of the matrix 
	@ X: input training data
	@ estimates: list of tuples that contain parameter estimates
	tuples are in the form (class,pi,mean,variance)
	"""

	# list of column vectors that contain bayes (log) probabilities for each class
	# we will eventually concatenate the output and predict the class that
	# has the highest probability

	bayes_probabilities = []

	# iterate through all estimates (which represents estimate for each class)
	# recall that each estimate is in in the form (class,pi,mean,variance)

	for estimate in estimates:
		pi = estimate[1]
		mean = estimate[2]
		variance = estimate[3]
		log_variance = np.log(variance)
		# variance inverse 
		sigma_inv = inv(log_variance)

		# use a for loop and add the bayes probabilities one by one
		# bayes_probs represents a single column vector
		bayes_probs = []
		for row in X:
			# make it a column vector
			x = row.reshape(-1,1)
			# calculate bayes prob for one entry
			# using the QDA formula
			bayes_prob = (-.5 * multi_dot([(x-mean).T,(sigma_inv),(x-mean)])[0][0]) - (.5 * np.log(det(log_variance))) + np.log(pi)

			bayes_probs.append(bayes_prob)

		bayes_probabilities.append(np.array(bayes_probs).reshape(-1,1))

	# now we will concatenate the probabilities for each class
	# and take the argmax, to find the index that had the highest
	# log probability.

	# for example, if the 3rd log probability (at index 2) of the first row 
	# was the highest, then the first entry of this array will contain a '2'

	indices_of_highest_prob = np.argmax(np.concatenate(bayes_probabilities,axis=1),axis=1)

	# now predict the class based on the index of the highest log probability.
	# for example, if the index was '1', this means that the log probability was
	# highest for the second set of estimates, and so we predict the class assigned
	# to that estimate (this is why we included the class in the tuple!)

	def predict_class(index):
		# the class is in the 0th index of the tuple
		return estimates[index][0]

	# create a function that does this to a vector
	predict_class_vec = np.vectorize(predict_class)


	predictions = predict_class_vec(indices_of_highest_prob)

	return predictions











