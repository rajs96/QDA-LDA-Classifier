import numpy as np
# allows us to multiply more than one vector at once
from numpy.linalg import multi_dot
# allows us to compute inverse of matrix
from numpy.linalg import inv


def LDA_classifier(X,estimates,variance_estimate):
	"""
	A function to return LDA classification output for a given X
	We use a vectorized implementation so that we can avoid
	for loops and speed up computation time
	@ X: input training data
	@ estimates: list of tuples that contain parameter estimates
	tuples are in the form (class,pi,mean,variance) BUT WE DISREGARD the variance
	@ variance_estimate: the variance estimate we use for the classifier
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
		# variance inverse 
		sigma_inv = inv(variance_estimate)

		# formula for linear discriminant
		# the second and third terms are BROADCASTED across the first term, which is a vector
		# with shape (# of observations, # of features)
		bayes_prob = multi_dot([X,sigma_inv,mean]) - (.5 * multi_dot([mean.T,sigma_inv,mean])) + np.log(pi)

		# appends a 
		bayes_probabilities.append(bayes_prob)

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











