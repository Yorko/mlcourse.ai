import numpy as np


def svm_loss_naive(w, x, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    :param w: A numpy array of shape (D, C) containing weights.
    :param x: A numpy array of shape (N, D) containing a mini-batch of data.
    :param y: A numpy array of shape (N,) containing training labels; y[i] = c means
              that x[i] has label c, where 0 <= c < C.
    :param reg: (float) regularization strength
    :return: Returns a tuple of:
             - loss as single float
             - gradient with respect to weights W; an array of same shape as W
    """

    dw = np.zeros(w.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = w.shape[1]
    num_train = x.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = x[i].dot(w)
        correct_class_score = scores[y[i]]
        # we'll count the number of classes that didn’t meet the desired margin
        num_positive_margin = 0
      
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # if the current class does't meet the desired margin, remember it
                num_positive_margin += 1
                # if the current class does't meet the desired margin, increment the gradient w.r.t.
                # to the current incorrect class by X_i
                dw[:, j] += x[i, :]
        # decrement the gradient w.r.t. to the correct class by X_i times the number of times other
        # classes didn’t meet the desired margin
        dw[:, y[i]] -= num_positive_margin * x[i, :]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dw /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(w * w)
    dw += 2 * reg * w

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################

    return loss, dw


def svm_loss_vectorized(w, x, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    num_train = x.shape[0]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # shape (num_train, num_classes)
    scores = x.dot(w)
    # select in each row i the score at position y[i]
    correct_class_scores = scores[range(num_train), y].reshape(-1, 1)
    # compute margins row-wise, shape (num_train, num_classes)
    margins = scores - correct_class_scores + 1
    # compute SVM loss
    loss = margins.clip(min=0).sum() - num_train
    loss /= num_train
    loss += reg * np.sum(w * w)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    # where the margins are positive, shape (num_train, num_classes)
    idx_positive_margins = np.greater(margins, 0).astype('int')
    # for each object and correct class substract the number of positive margins
    idx_positive_margins[range(num_train), y] -= idx_positive_margins.sum(axis=1)
    # multiple X transposed by the previous matrix
    dw = x.T.dot(idx_positive_margins)
    dw /= num_train
    dw += 2 * reg * w
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dw
