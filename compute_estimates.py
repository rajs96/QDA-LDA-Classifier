import numpy as np

def compute_estimates(X_train,y_train,classifier='QDA'):
	""" Function to compute estimates for a LDA or QDA classifier"""
	# get a list of the different classes
	classes = list(np.unique(y_train))

	# list of tuples that contain estimates for each class
	# tuple is in the form (class,pi,mean,variance)
	estimates = []

	for c in classes:
		# have it as a list originally,
		# then turn it into a tuple
		estimate = []

		# add the class as the first element of the tuple
		estimate.append(c)

		# first we want to subset the data for that particular class
		# get the indices of the rows for this particular class
		indices_of_rows = np.where(np.isin(y_train,c))
		X_train_subset = X_train[indices_of_rows]

		pi = float(len(X_train_subset))/float(len(X_train))
		estimate.append(pi)

		# reshape makes it a proper column vector
		mean = (np.sum(X_train_subset,axis=0) / float(len(X_train_subset))).reshape(-1,1)
		estimate.append(mean)
		
		def take_cov(row,mean):
			""" 
			Function that takes in an observation and the mean
			@row: observation vector (not reshaped yet)
			@mean: mean vector that HAS BEEN RESHAPED
			"""

			return (row.reshape(-1,1) - mean).dot((row.reshape(-1,1) - mean).T)

		# do a list comprehension to sum over individual variances
		# to get a variance vector 
		variance = (1./(len(X_train_subset) - len(classes))) * (sum([take_cov(row,mean) for row in X_train_subset]))
		estimate.append(variance)
		estimates.append(tuple(estimate))

	# need to add the variance matrices if we have LDA
	# and make this the variance for every class
	if classifier == 'LDA':
		# estimate[3] represents the variance
		variance = sum([estimate[3] for estimate in estimates])

		# return a tuple with list of estimates
		# along with the estimator for the variance
		# remember var_class1 = var_class2 = ... = var_classn
		# for LDA
		return (estimates,variance)

	return estimates



