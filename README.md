# QDA/LDA Classifier

Here, we have two programs: one that uses linear discriminant analysis to implement a bayes classifier, and one that uses quadratic discriminant analysis. Note that LDA is the same as QDA, with the exception that variance matrices for each class are the same. Because we obviously don't know the parameters (i.e. the mean and variance) for the conditional distribution of each class, we used unbiased estimators. Many of the implementation details can be found in section 4.3 of "Elements of Statistical Learning."

There is also a test script titled "test.py" that uses a dataset of handwritten digits. We test the classifier on a subset of the data, namely those that are classified as "3" and "5". When we test the QDA classifier, the individual variance matrices are not invertible, and so Python reports incorrect results. Do not worry when using it, the QDA classifier will work properly as long as the variance matrices are invertible for each class in the dataset.

Looking at the in the Python scripts is really helpful in understanding the code.