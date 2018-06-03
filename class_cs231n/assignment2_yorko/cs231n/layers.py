import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    out = x.reshape(x.shape[0], -1).dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = dout.sum(axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(x, 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout.copy()
    dx[x < 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from mini-batch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    n, d = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(d, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(d, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################

        # 1. calculate mean
        mu = np.mean(x, axis=0)

        # 2. subtract mean from every column
        x_centered = x - mu

        # 3. the same, squared
        x_centered_sq = x_centered ** 2

        # 4. calculate variance
        var = np.mean(x_centered_sq, axis=0)

        # 5. take sqrt, first adding eps (just in case var=0 not to divide by 0)
        std = np.sqrt(var + eps)

        # 6. invert std
        inv_std = 1. / std

        # 7. normalization
        x_normed = x_centered * inv_std

        # 8. rescale
        x_normed_rescaled = gamma * x_normed

        # 9. shift
        out = x_normed_rescaled + beta

        # 10. cache almost everything for backward pass
        cache = (x, mu, x_centered, var, std, inv_std, x_normed, gamma, eps)

        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################

        inv_std = 1. / np.sqrt(running_var + eps)
        out = (x - running_mean) * inv_std * gamma + beta

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(grad_out, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - grad_out: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - grad_gamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - grad_beta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################

    # 10. unpack cache
    x, mu, x_centered, var, std, inv_std, x_normed, gamma, eps = cache
    n, d = grad_out.shape

    # 9. forward was: out = x_normed_rescaled + beta
    grad_x_normed_rescaled = grad_out
    grad_beta = np.sum(grad_out, axis=0)

    # 8. forward was:  x_normed_rescaled = gamma * x_normed
    grad_x_normed = grad_x_normed_rescaled * gamma
    grad_gamma = np.sum(grad_out * x_normed, axis=0)

    # 7. forward was: x_normed = x_centered * inv_std
    grad_x_centered1 = grad_x_normed * inv_std
    grad_inv_std = np.sum(grad_x_normed * x_centered, axis=0)

    # 6. forward was: inv_std = 1. / std
    grad_std = grad_inv_std * (-1. / std ** 2)

    # 5. forward was: std = np.sqrt(var + eps)
    grad_var = grad_std * (1. / (2 * std))

    # 4. forward was: var = np.mean(x_centered_sq, axis=0)
    grad_x_centered_sq = grad_var * (1. / n * np.ones((n, d)))

    # 3. forward was: x_centered_sq = x_centered ** 2
    grad_x_centered2 = grad_x_centered_sq * 2 * x_centered
    grad_x_centered = grad_x_centered1 + grad_x_centered2

    # 2. forward was: x_centered = x - mu
    grad_x1 = grad_x_centered
    grad_mu = - np.sum(grad_x_centered, axis=0)

    # 1. mu = np.mean(x, axis=0)
    grad_x2 = grad_mu * (1. / n * np.ones((n, d)))
    grad_x = grad_x1 + grad_x2

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return grad_x, grad_gamma, grad_beta


def batchnorm_backward_alt(grad_out, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalization backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # http://cthorey.github.io/backpropagation
    # https://kevinzakka.github.io/2016/09/14/batch_normalization/
    x, mu, x_centered, var, std, inv_std, x_normed, gamma, eps = cache
    n, d = grad_out.shape

    grad_gamma = (grad_out * x_normed).sum(axis=0)
    grad_beta = grad_out.sum(axis=0)

    # http://cthorey.github.io./backpropagation/
    grad_x = (1. / n) * gamma * inv_std * (n * grad_out - grad_out.sum(axis=0) -
                                           x_centered * inv_std ** 2 * (grad_out * x_centered).sum(axis=0))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return grad_x, grad_gamma, grad_beta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    sample_mean = x.mean(axis=1, keepdims=True)  # shape (N, 1)
    sample_var = x.var(axis=1, keepdims=True)  # shape (N, 1)
    x_normed = (x - sample_mean) / np.sqrt(sample_var + eps)  # shape (N, D)
    out = x_normed * gamma + beta  # shape (N, D)
    cache = (sample_mean, sample_var, x_normed, gamma, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(grad_out, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    sample_mean, sample_var, x_normed, gamma, eps = cache
    grad_beta = grad_out.sum(axis=0)
    grad_gamma = (x_normed * grad_out).sum(axis=0)

    grad_x_normed = grad_out * gamma
    std = np.sqrt(sample_var + eps)
    grad_mu = - grad_x_normed.sum(axis=1, keepdims=True) / std
    grad_var = - (grad_x_normed * x_normed).sum(axis=1, keepdims=True) / (sample_var + eps) / 2
    d = grad_out.shape[1]
    grad_x = grad_x_normed / std + grad_mu / d + x_normed * std * grad_var * 2 / d
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return grad_x, grad_gamma, grad_beta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(grad_out, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - grad_out: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        grad_x = grad_out * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        grad_x = grad_out
    return grad_x


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    padding, stride = conv_param['pad'], conv_param['stride']

    n, c, image_height, image_width = x.shape
    f, _, filter_height, filter_width = w.shape

    output_height = (image_height - filter_height + 2 * padding) // stride + 1
    output_width = (image_width - filter_width + 2 * padding) // stride + 1

    out = np.zeros((n, f, output_height, output_width))
    w_flattened = w.reshape((f, -1))

    # padding
    x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), 'constant')
    for i in range(output_height):
        for j in range(output_width):
            x_slice = x[:, :, i * stride: i * stride + filter_height, j * stride: j * stride + filter_width]
            x_slice_flattened = x_slice.reshape((n, -1))
            out_slice = x_slice_flattened.dot(w_flattened.T) + b
            out[:, :, i, j] = out_slice
            ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(grad_out, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - grad_out: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - grad_x: Gradient with respect to x
    - grad_w: Gradient with respect to w
    - grad_b: Gradient with respect to b
    """
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    padding, stride = conv_param['pad'], conv_param['stride']

    f, c, filter_height, filter_width = w.shape
    n, _, output_height, output_width = grad_out.shape

    grad_x = np.zeros(x.shape)
    grad_w = np.zeros(w.shape)
    grad_b = np.zeros(b.shape)

    w_flat = w.reshape((f, -1))

    # If in Russian, see habr.com papers
    # https://habr.com/company/ods/blog/344008/
    # https://habr.com/company/ods/blog/344116/
    # https://habr.com/company/ods/blog/344888/

    for i in range(output_height):
        for j in range(output_width):
            grad_out_slice = grad_out[:, :, i, j]

            grad_x_slice_flattened = grad_out_slice.dot(w_flat)
            grad_x_slice = grad_x_slice_flattened.reshape((n, c, filter_height, filter_width))
            grad_x[:, :, i * stride: i * stride + filter_height, j * stride: j * stride + filter_width] += grad_x_slice

            x_slice = x[:, :, i * stride: i * stride + filter_height, j * stride: j * stride + filter_width]
            x_slice_flattened = x_slice.reshape((n, -1))

            grad_w += grad_out_slice.T.dot(x_slice_flattened).reshape(grad_w.shape)
            grad_b += grad_out_slice.sum(axis=0)

    # crop padding from grad_x
    grad_x = grad_x[:, :, padding:-padding, padding:-padding]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return grad_x, grad_w, grad_b


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    stride = pool_param['stride']
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']

    n, c, image_height, image_width = x.shape

    output_height = (image_height - pool_height) // stride + 1
    output_width = (image_width - pool_width) // stride + 1

    out = np.zeros((n, c, output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            # resembles convolution very much
            x_slice = x[:, :, i * stride: i * stride + pool_height, j * stride: j * stride + pool_width]
            # here we take maximal values along geometrical dimensions (height and width)
            out[:, :, i, j] = np.amax(x_slice, axis=(2, 3))
            ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(grad_out, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - grad_out: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - grad_x: Gradient with respect to x
    """
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    stride = pool_param['stride']
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']

    n, c, output_height, output_width = grad_out.shape

    grad_x = np.zeros(x.shape)

    for i in range(output_height):
        for j in range(output_width):
            x_slice = x[:, :, i * stride: i * stride + pool_height, j * stride: j * stride + pool_width]
            for n_iterator in range(n):
                for c_iterator in range(c):
                    max_activation = np.amax(x_slice[n_iterator, c_iterator, :, :])
                    for k in range(pool_height):
                        for m in range(pool_width):
                            if x[n_iterator, c_iterator, i * stride + k, j * stride + m] == max_activation:
                                grad_x[n_iterator, c_iterator, i * stride + k, j * stride + m] += \
                                    grad_out[n_iterator, c_iterator, i, j]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return grad_x


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
