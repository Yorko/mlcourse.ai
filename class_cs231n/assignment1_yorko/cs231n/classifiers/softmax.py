import numpy as np


def softmax_loss_naive(w, x, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    :param w: A numpy array of shape (D, C) containing weights.
    :param x: A numpy array of shape (N, D) containing a minibatch of data.
    :param y: A numpy array of shape (N,) containing training labels; y[i] = c means
              that x[i] has label c, where 0 <= c < C.
    :param reg: (float) regularization strength
    :return: Returns a tuple of:
             - loss as single float
             - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dw = np.zeros_like(w)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_classes = w.shape[1]
    num_train = x.shape[0]
    for i in range(num_train):
        # classification scores for the current sample, shape (1, num_classes)
        scores = x[i].dot(w)
        # just a scalar value
        correct_class_score = scores[y[i]]
        # applying the log-sum-exp trick
        # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        max_score = scores.max()
        scores -= max_score
        # compute softmax loss for the current sample
        loss += -correct_class_score + max_score + np.log(np.exp(scores).sum())
        # compute gradients w.r.t. to weights for each class separately
        for j in range(num_classes):
            dw[:, j] += np.exp(scores[j]) / np.exp(scores).sum() * x[i, :]
        # the formula for gradients w.r.t. to the weights of the correct class
        # is slightly different http://cs231n.github.io/optimization-1/
        dw[:, y[i]] -= x[i, :]

    loss /= num_train
    loss += reg * np.sum(w * w)

    dw /= num_train
    dw += 2 * reg * w
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dw


def softmax_loss_vectorized(w, x, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train = x.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = x.dot(w)
    # select in each row i the score at position y[i]
    # the formula is given here http://cs231n.github.io/linear-classify/#softmax
    correct_class_scores = scores[range(num_train), y]
    # applying the log-sum-exp trick
    # https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
    max_scores = scores.max(axis=1, keepdims=True)
    scores -= max_scores
    # compute softmax loss
    loss = - correct_class_scores.sum() + max_scores.sum() + np.log(np.exp(scores).sum(axis=1)).sum()
    loss /= num_train
    loss += reg * np.sum(w * w)

    # compute softmax derivatives w.r.t. to scores, shape (num_train, num_classes)
    softmax_deriv = (np.exp(scores) / np.exp(scores).sum(axis=1).reshape(-1, 1))
    softmax_deriv[range(num_train), y] -= 1
    # compute softmax gradients w.r.t. to weights W, shape (num_features, num_classes)
    dw = x.T.dot(softmax_deriv)
    dw /= num_train
    dw += 2 * reg * w
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dw
