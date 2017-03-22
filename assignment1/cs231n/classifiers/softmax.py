import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = np.shape(X)[0]
  num_class = np.shape(W)[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in range(num_train):
    scores = W.T.dot(X[i,:].T)
    scores = np.exp(scores)/np.sum(np.exp(scores))
    loss+= -np.log(scores[y[i]])

    #Gradients
    ## when the class is correct (i.e i when yi==c) gradient is probi-1 multiplied
    # by the X values corespinding to each weigh
    ## when the class is not the correct class, gradient is probj multiplied by
    # corresponding X values
    for j in range(num_class):
      dW[:,j] += scores[j]*X[i,:].T
      if j == y[i]: # accounting for the 1 when the class is correct
        dW[:, j] -= X[i, :].T

  loss /= num_train
  dW /= num_train

  #add regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = np.shape(X)[0]
  num_class = np.shape(W)[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #numerical stbility
  scores -= np.amax(scores, axis= 1, keepdims= True)
  scores = np.exp(scores)
  scores /= np.sum(scores, axis = 1, keepdims=True)
  loss = -np.sum(np.log(scores[np.arange(num_train),y]))
  loss /= num_train
  scores[np.arange(num_train),y]-=1
  dW += X.T.dot(scores)
  dW/=num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W






  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

