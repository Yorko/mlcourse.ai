import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, x, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        b1, b2 = self.params['b1'], self.params['b2']
        w1, w2 = self.params['W1'], self.params['W2']
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        hidden, cache_hidden = affine_relu_forward(x, w1, b1)
        scores, cache_out = affine_forward(hidden, w2, b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, grad_loss = softmax_loss(scores, y)
        loss += self.reg * (np.sum(w1 * w1) + np.sum(w2 * w2)) / 2

        grad_hidden, grad_w2, grad_b2 = affine_backward(grad_loss, cache_out)
        _, grad_w1, grad_b1 = affine_relu_backward(grad_hidden, cache_hidden)

        grads['W1'] = grad_w1 + self.reg * w1
        grads['W2'] = grad_w2 + self.reg * w2
        grads['b1'], grads['b2'] = grad_b1, grad_b2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        all_dims = [input_dim] + hidden_dims + [num_classes]
        for layer_id in range(1, self.num_layers + 1):
            w = weight_scale * np.random.randn(all_dims[layer_id - 1], all_dims[layer_id])
            b = np.zeros(all_dims[layer_id])
            self.params['W%d' % layer_id], self.params['b%d' % layer_id] = w, b

            if (self.normalization == 'batchnorm' or self.normalization == 'layernorm') and layer_id < self.num_layers:
                self.params['gamma%d' % layer_id] = np.ones(all_dims[layer_id])
                self.params['beta%d' % layer_id] = np.zeros(all_dims[layer_id])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for _ in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, x, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        x = x.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        current_x = x
        caches = {}

        # loop over L-1 cycles : {affine - [batch norm] - relu - [dropout]} x (L - 1)
        for layer_id in range(1, self.num_layers):
            # forward pass through current affine layer
            current_x, cache = affine_forward(current_x, self.params['W%d' % layer_id],
                                              self.params['b%d' % layer_id])

            # save cache for backward pass
            caches['affine%d' % layer_id] = cache

            if self.normalization == 'batchnorm':
                current_x, cache = batchnorm_forward(current_x,
                                                     self.params['gamma%d' % layer_id],
                                                     self.params['beta%d' % layer_id],
                                                     self.bn_params[layer_id - 1])
                caches['batchnorm%d' % layer_id] = cache
            elif self.normalization == 'layernorm':
                current_x, cache = layernorm_forward(current_x,
                                                     self.params['gamma%d' % layer_id],
                                                     self.params['beta%d' % layer_id],
                                                     self.bn_params[layer_id - 1])
                caches['layernorm%d' % layer_id] = cache

            # forward pass through current ReLU layer and saving ReLU cache
            current_x, cache = relu_forward(current_x)
            caches['relu%d' % layer_id] = cache

            if self.use_dropout:
                current_x, cache = dropout_forward(current_x, self.dropout_param)
                caches['dropout%d' % layer_id] = cache

        # final affine layer
        scores, cache_out = affine_forward(current_x, self.params['W%d' % self.num_layers],
                                           self.params['b%d' % self.num_layers])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        # calculating softmax loss with regularizers for weights in all layers
        loss, grad_loss = softmax_loss(scores, y)
        for layer_id in range(self.num_layers):
            loss += self.reg * np.sum(self.params['W%d' % (layer_id + 1)] ** 2) / 2

        # backward pass through final affine layer
        grad_current_x, dw, db = affine_backward(grad_loss, cache_out)
        grads['W%d' % self.num_layers] = dw + self.reg * self.params['W%d' % self.num_layers]
        grads['b%d' % self.num_layers] = db

        # loop over L-1 cycles : {affine - [batch norm] - relu - [dropout]} x (L - 1)
        for layer_id in range(self.num_layers - 1, 0, -1):
            if self.use_dropout:
                grad_current_x = dropout_backward(grad_current_x, caches['dropout%d' % layer_id])

            # backward pass through current ReLU layer
            grad_current_x = relu_backward(grad_current_x, caches['relu%d' % layer_id])

            if self.normalization == 'batchnorm':
                grad_current_x, grad_gamma, grad_beta = batchnorm_backward(grad_current_x,
                                                                           caches['batchnorm%d' % layer_id])
                grads['gamma%d' % layer_id] = grad_gamma
                grads['beta%d' % layer_id] = grad_beta

            elif self.normalization == 'layernorm':
                grad_current_x, grad_gamma, grad_beta = layernorm_backward(grad_current_x,
                                                                           caches['layernorm%d' % layer_id])
                grads['gamma%d' % layer_id] = grad_gamma
                grads['beta%d' % layer_id] = grad_beta

            # backward pass through current affine layer
            grad_current_x, dw, db = affine_backward(grad_current_x, caches['affine%d' % layer_id])
            grads['W%d' % layer_id] = dw + self.reg * self.params['W%d' % layer_id]
            grads['b%d' % layer_id] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
