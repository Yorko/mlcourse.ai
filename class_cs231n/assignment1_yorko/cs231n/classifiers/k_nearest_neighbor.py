import numpy as np


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.x_train = None
        self.y_train = None

    def train(self, x, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        :param x: A numpy array of shape (num_train, D) containing the training data
                  consisting of num_train samples each of dimension D.
        :param y: A numpy array of shape (N,) containing the training labels, where
                  y[i] is the label for X[i].
        :return: None
        """

        self.x_train = x
        self.y_train = y
    
    def predict(self, x, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        :param x: A numpy array of shape (num_test, D) containing test data consisting
                  of num_test samples each of dimension D.
        :param k: The number of nearest neighbors that vote for the predicted labels.
        :param num_loops: Determines which implementation to use to compute distances
                          between training points and testing points.
        :return y: A numpy array of shape (num_test,) containing predicted labels for the
                   test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(x)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(x)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(x)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, x):
        """
        Compute the distance between each test point in X and each training point
        in self.x_train using a nested loop over both the training data and the
        test data.

        :param x: A numpy array of shape (num_test, D) containing test data.
        :return dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                       is the Euclidean distance between the ith test point and the
                       jth training point.
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for test_id in range(num_test):
            for train_id in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension.                                    #
                #####################################################################

                dists[test_id, train_id] = np.linalg.norm(x[test_id, :]
                                                          - self.x_train[train_id, :])

                #####################################################################
                #                       END OF YOUR CODE                            #
                #####################################################################
        return dists

    def compute_distances_one_loop(self, x):
        """
        Compute the distance between each test point in X and each training point
        in self.x_train using a single loop over the test data.

        :param x: A numpy array of shape (num_test, D) containing test data.
        :return dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                      is the Euclidean distance between the ith test point and the
                      jth training point.
        """
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for test_id in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            #######################################################################
            
            # just use the argument "axis"
            dists[test_id, :] = np.linalg.norm(x[test_id, :] - self.x_train, axis=1)

            #######################################################################
            #                         END OF YOUR CODE                            #
            #######################################################################
        return dists

    def compute_distances_no_loops(self, x):
        """
        Compute the distance between each test point in X and each training point
        in self.x_train using no explicit loops.

        :param x: A numpy array of shape (num_test, D) containing test data.
        :return dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                      is the Euclidean distance between the ith test point and the
                      jth training point.
        """
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy.                #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        # shape (num_test, 1)
        x_norms = np.sum(x ** 2, axis=1).reshape(-1, 1)
        # shape (num_train, )
        x_train_norms = np.sum(self.x_train ** 2, axis=1)
        # summing arrays of shapes (num_test, 1) and (num_train, )
        # gives exactly the necessary shape (num_test, num_train). It's broadcasting.
        dists = (x_norms + x_train_norms - 2 * x.dot(self.x_train.T)) ** .5

        #########################################################################
        #                         END OF YOUR CODE                              #
        #########################################################################
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        :param dists: A numpy array of shape (num_test, num_train) where dists[i, j]
                      gives the distance between the ith test point and the jth training point.
        :param k: number of neighbors in kNN (hyper-parameter)
        :return y: A numpy array of shape (num_test,) containing predicted labels for the
                   test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################

            closest_idx = np.argsort(dists[i, :])[:k]

            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################

            neighbor_labels = self.y_train[closest_idx]
            label_counts = np.bincount(neighbor_labels)
            y_pred[i] = np.argmax(label_counts)

            #########################################################################
            #                           END OF YOUR CODE                            # 
            #########################################################################

        return y_pred
