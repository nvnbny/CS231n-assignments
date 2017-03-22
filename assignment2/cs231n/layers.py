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
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  N = np.shape(x)[0]
  out = np.reshape(x,(N,-1)).dot(w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  dx, dw, db = None, None, None
  N = np.shape(x)[0]
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # dx
  dx = dout.dot(w.T)
  dx = np.reshape(dx,x.shape)
  # dw
  dw = np.reshape(x,(N,-1)).T.dot(dout)
  # db
  db = np.sum(dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0,x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = (x>0)*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
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

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    # Step 1
    mu = np.mean(x, axis=0) # shape (D,)

    # Step 2
    xmu = x - mu # Shape (N,D)

    # Step 3
    sqrs = xmu**2 # Shape (N,D)

    # Step 4
    var = np.sum(sqrs, axis=0)/N + eps # Shape (D,)

    # Step 5
    sqrtvar = np.sqrt(var) # Shape (D,)

    # Step 6
    insqrtvar = 1.0/sqrtvar # Shape (D,)

    # Step 7
    xhat = xmu * insqrtvar # Shape (N,D)

    # Step 8
    xgamma = xhat * gamma

    # Step 9
    out = xgamma + beta

    cache = (mu, xmu, sqrs, var, sqrtvar, insqrtvar, xhat, xgamma, out, gamma, beta, x, bn_param)

    running_mean = momentum * mu + (1 - momentum) * running_mean
    running_var = momentum * var + (1 - momentum) * running_var

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    out = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * out + beta

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """
  dx, dgamma, dbeta = None, None, None
  mu, xmu, sqrs, var, sqrtvar, insqrtvar, xhat, xgamma, out, gamma, beta, x, bn_param = cache
  N, D = np.shape(dout)
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  # Step 9
  # out = xgamma + beta # shape (N,D)
  dbeta = np.sum(dout, axis=0)
  dxgamma = dout

  # Step 8
  # xgamma = xhat * gamma #shape (N,D)
  dgamma = np.sum(xhat * dout, axis=0)
  dxhat = gamma * dxgamma

  # Step 7
  # xhat = xmu * insqrtvar  # Shape (N,D)
  dxmu = insqrtvar * dxhat
  dinsqrtvar = np.sum(xmu * dxhat, axis=0)

  # Step 6
  # insqrtvar = 1.0 / sqrtvar  # Shape (D,)
  dsqrtvar = (-1.0/(sqrtvar)**2) * dinsqrtvar

  # Step 5
  # sqrtvar = np.sqrt(var)  # Shape (D,)
  dvar = (0.5*var**-0.5) * dsqrtvar

  # Step 4
  # var = np.sum(sqrs, axis=0) / N + eps # Shape (D,)
  dsqrs = np.ones(np.shape(sqrs)) * 1./N * dvar

  # Step 3
  # sqrs = xmu ** 2  # Shape (N,D)
  dxmu += 2 * xmu * dsqrs

  # Step 2
  # xmu = x - mu  # Shape (N,D)
  dx = dxmu
  dmu = np.sum(-1 * dxmu, axis=0)

  # Step 1
  # mu = np.mean(x, axis=0)  # shape (D,)
  dx += 1./N * np.ones((np.shape(dxmu))) * dmu

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################


  mu, xmu, sqrs, var, sqrtvar, insqrtvar, xhat, xgamma, out, gamma, beta, x, bn_param = cache
  N, D = np.shape(dout)

  # intermediate partial derivatives
  dxhat = dout * gamma
  #final partial derivatives
  dx = (1. / N) * insqrtvar * (N * dxhat - np.sum(dxhat, axis=0)
                                - xhat * np.sum(dxhat * xhat, axis=0))
  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(xhat * dout, axis=0)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    # create a mask of shape X and random values from 0 to 1
    # make values below p to 1 and devide by p
    mask = (np.random.rand(*np.shape(x)) < p)/p

    # multiply mask and x
    out = x * mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = mask * dout

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

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
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################

  # essential variables
  N, C, H, W = np.shape(x)
  F, C, HH, WW = np.shape(w)
  pad = conv_param['pad']
  stride = conv_param['stride']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride

  # Padding
  x = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')

  out = np.zeros((N, F, H_out, W_out))
  for n in range(N): # every image
      for f in range(F): # every filter
        for h_out in range(H_out): # for every vertical stride
          for w_out in range(W_out): # for every horizontal stride
            out[n, f, h_out, w_out] = np.sum(w[f, :, :, :] * x[n, :, (h_out*stride):((h_out * stride) + HH), (w_out*stride):((w_out * stride) + WW)]) + b[f]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  H_out, W_out = np.shape(dout[0,0])
  N, C, H, W = np.shape(x)
  F, C, HH, WW = np.shape(w)
  pad = conv_param['pad']
  stride = conv_param['stride']

  db = np.sum(dout, axis=(0,2,3))
  dx= np.zeros_like(x)
  dw = np.zeros_like(w)

  # Calculating dx
  for n in range(N): # every image
      for f in range(F): # every filter
        for h_out in range(H_out): # for every vertical stride
          for w_out in range(W_out): # for every horizontal stride
            dx[n, :, (h_out*stride):((h_out * stride) + HH), (w_out*stride):((w_out * stride) + WW)] += w[f, :, :, :] * dout[n,f,h_out, w_out]
  # calculating dw
  for n in range(N):  # every image
    for f in range(F):  # every filter
      for h_out in range(H_out):  # for every vertical stride
        for w_out in range(W_out):  # for every horizontal stride
          dw[f, :, :, :] += x[n, :, (h_out * stride):((h_out * stride) + HH), (w_out * stride):((w_out * stride) + WW)] * dout[n, f, h_out, w_out]

  #remove padding
  dx = dx[:, :, 1:-1, 1:-1]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


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
  N,C,H,W = np.shape(x)
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  H_out =  1 + (H - HH)/stride
  W_out = 1 + (W - WW) / stride
  out = np.empty([N,C,H_out,W_out])

  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################

  for n in range(N):
    for c in range(C):
      for h_out in range(H_out):
        for w_out in range(W_out):
          out[n, c ,h_out, w_out] = np.max(x[n, c, (h_out * stride):(h_out*stride + HH), (w_out * stride):(w_out*stride + WW)])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """

  x, pool_param = cache
  N, C, H, W = np.shape(x)
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  H_out = 1 + (H - HH) / stride
  W_out = 1 + (W - WW) / stride
  dx = np.zeros_like(x)

  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################

  for n in range(N):
    for c in range(C):
      for h_out in range(H_out):
        for w_out in range(W_out):
          sub_x = x[n, c, (h_out * stride):(h_out*stride + HH), (w_out * stride):(w_out*stride + WW)]
          i, j = np.unravel_index(sub_x.argmax(), sub_x.shape)
          dx[n, c, (h_out * stride) +i, (w_out * stride) + j] += dout[n, c , h_out, w_out]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


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


  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  N, C, H, W = x.shape

  # manipulate the axes and shape
  x = np.swapaxes(x, 1, 3) # swap the filter and width axes first
  x = np.swapaxes(x, 1, 2) # swap the height and width
  new_shape = x.shape
  x = np.reshape(x, (-1, C)) # reshape to proper N x D

  out, cache = batchnorm_forward( x, gamma, beta, bn_param)

  #back to previous shape
  out = np.reshape(out, new_shape)
  out = np.swapaxes(out, 1, 2)
  out = np.swapaxes(out, 1, 3)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

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

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################


  N, C, H, W = dout.shape

  # manipulate the axes and shape
  dout = np.swapaxes(dout, 1, 3) # swap the filter and width axes first
  dout = np.swapaxes(dout, 1, 2) # swap the height and width
  new_shape = dout.shape
  dout = np.reshape(dout, (-1, C)) # reshape to proper N x D

  dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)

  #back to previous shape
  dx = np.reshape(dx, new_shape)
  dx = np.swapaxes(dx, 1, 2)
  dx = np.swapaxes(dx, 1, 3)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
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
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
