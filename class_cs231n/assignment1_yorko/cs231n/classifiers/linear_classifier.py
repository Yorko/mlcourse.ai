from .linear_svm import *
from .softmax import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, x, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        :param x: A numpy array of shape (N, D) containing training data; there are N
                  training samples each of dimension D.
        :param y: A numpy array of shape (N,) containing training labels; y[i] = c
                  means that X[i] has label 0 <= c < C for C classes.
        :param learning_rate: (float) learning rate for optimization.
        :param reg: (float) regularization strength.
        :param num_iters: (integer) number of steps to take when optimizing
        :param batch_size: (integer) number of training examples to use at each step.
        :param verbose: (boolean) If true, print progress during optimization.
        :return loss_history: A list containing the value of the loss function at each training iteration.
        """

        num_train, dim = x.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []

        for it in range(num_iters):
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            idx = np.random.choice(range(x.shape[0]), size=batch_size)
            x_batch, y_batch = x[idx, :], y[idx]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(x_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO: Update the weights using the gradient and the learning rate.    #
            #########################################################################
            self.W -= learning_rate * grad
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, x):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        :param x: A numpy array of shape (N, D) containing training data; there are N
                  training samples each of dimension D.
        :return y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
                        array of length N, and each element is an integer giving the
                        predicted class.
        """
        ###########################################################################
        # TODO: Implement this method. Store the predicted labels in y_pred.      #
        ###########################################################################
        y_pred = np.argmax(x.dot(self.W), axis=1)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, x_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        :param x_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        :param y_batch: A numpy array of shape (N,) containing labels for the mini-batch.
        :param reg: (float) regularization strength.
        :return: A tuple containing:
                     - loss as a single float
                     - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, x_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, x_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, x_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, x_batch, y_batch, reg)
