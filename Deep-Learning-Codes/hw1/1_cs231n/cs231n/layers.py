import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
#  x = x.reshape(len(x), -1)
  out = np.dot(x.reshape(len(x), -1), w) + b
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
  #############################################################################
  # Implement the affine backward pass.                                 #
  #############################################################################
  dx = np.dot(dout, w.T).reshape(x.shape)
  dw = np.dot(x.reshape(len(x), -1).T, dout)
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
  # Implement the ReLU forward pass.                                    #
  #############################################################################
  out = x * (x>0)
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
  #  Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout * ((x>0)*1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
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
  #  Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
#  print(x)
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
#  print(N, C, H, W)
  stride, pad = conv_param['stride'], int(conv_param['pad'])
#  print(stride, pad)
  Ho = int(1 + (H + 2 * pad - HH) / stride)
  Wo = int(1 + (W + 2 * pad - WW) / stride)
  out = np.zeros((N, F, Ho, Wo))
  # we first create a copy of the input array with padding.. this is not efficient implementation 
  # as we are making a redundant copy just for the sake of convenience of not dealing with the indices
  pad_x = np.pad(x, ((0,),(0,),(pad,),(pad,)),'constant')
#  print(pad_x)
  # The leading dimensions of the x, w determine the leading dims of 'out'.
  # Note that the np.dot product gives NxF output and we want ixj such outputs (see i, j in for loops below)
  for i, i1 in enumerate(range(0, stride*Ho, stride)):# i1 -> moves over original image and i-> conv output
      for j, j1 in enumerate(range(0, stride*Wo, stride)):
          out[:, :, i, j] = (np.dot(pad_x[:, :, i1:i1+HH, j1:j1+WW].reshape(N, -1), w.reshape(F, -1).T)+b)#shape(NxF)
#  print(out)
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
  x, w, b, conv_param = cache
  dx, dw, db = None, None, None
  #############################################################################
  # Implement the convolutional backward pass.                          #
  #############################################################################
  # Retrieving the constants(sizes and stuff)
  N, F, Ho, Wo = dout.shape
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
#  print(N, C, H, W)
  stride, pad = conv_param['stride'], int(conv_param['pad'])
  # get the padded x, as convolution is performed on it  
  pad_x = np.pad(x, ((0,),(0,),(pad,),(pad,)), 'constant')
  dpad_x = np.zeros(pad_x.shape)
  dw = np.zeros(w.shape)
  db = np.zeros((F))
  # sum over its dimensions (we get all ones in F dimension then, we need to sum up dout in all the other dimensions for the derivatives )
  for i in range(F):
      db[i] = np.sum(dout[:, i, :, :]) # Could've done it without the for loop ...
  # it is just simply adding the derivatives of the convolution operations done before
  for i, i1 in enumerate(range(0, stride*Ho, stride)):
      for j, j1 in enumerate(range(0, stride*Wo, stride)):
#          dw[:, :, i, j] = np.dot(dout[:, :, i, j].T, pad_x[:, :, i1:i1+HH, j1:j1+WW])   
# Note: we just need to match the dimensions now... 
#dout[:, :, i, j]--> (NxF) and pad_x[:, :, i1:i1+HH, j1:j1+WW]--> (N x C x HH x WW)
# dw[:, :, i, j] --> FxCxHxW
          dw = dw + np.dot(dout[:, :, i, j].transpose(), pad_x[:, :, i1:i1+HH, j1:j1+WW].reshape(N, -1)).reshape(F, C, HH, WW)
# Similarly (dout times w filter) (NxF) times (F, C), we need (N, C,...), thus we need to multiply to eliminate the 'F' dimension
          dpad_x[:, :, i1:i1+HH, j1:j1+WW] = dpad_x[: ,:, i1:i1+HH, j1:j1+WW] + np.dot(dout[:, :, i, j], w.reshape(F, -1)).reshape(N, C, HH, WW)
# Removing the padding to get the original x, Not Sure about this step though
  dx = dpad_x[:, :, pad:pad+H, pad:pad+W]

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
  out = None
  #############################################################################
  # Implement the max pooling forward pass                              #
  #############################################################################
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  # input data dimensions
  N, C, H, W = x.shape
  # output shape
  Ho = int((H-pool_height)/stride + 1)
  Wo = int((W-pool_width)/stride + 1)
  out = np.zeros((N, C, Ho, Wo))
  #Now we again go through the complete set of images and all the channels and do the max pool 
  for n1 in range(N):# extremely naive implementation.. got lazy to figure out the indices and all
      for c1 in range(C):
          for h1 in range(Ho):
              for w1 in range(Wo):
                  # extract the receptive area
                  receptive_area = x[n1, c1, h1*stride:h1*stride+pool_height, w1*stride:w1*stride+pool_width]
                  out[n1, c1, h1, w1] = np.max(receptive_area)

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
  dx = None
  #############################################################################
  #  Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  N, C, H, W = x.shape
  N, C, Ho, Wo = dout.shape
  dx = np.zeros(x.shape)
  #Now we again go through the complete set of images and all the channels and do the max pool derivatives 
  for n1 in range(N):# extremely naive implementation.. got lazy to figure out the indices and all
      for c1 in range(C):
          for h1 in range(Ho):
              for w1 in range(Wo):
                  #indices_update_region = [n1, c1, h1*stride:h1*stride+pool_height, w1*stride:w1*stride+pool_width]
                  receptive_area = x[n1, c1, h1*stride:h1*stride+pool_height, w1*stride:w1*stride+pool_width]
                  # elementwise multiplication for letting the max element from the input derivatives pass back
                  update_dx = dout[n1, c1, h1, w1]*((receptive_area == np.max(receptive_area))*1)
                  # Now, we need to find out the position of the max element in the receptive area and 
                  # the derivative of the same will be equal to 1. Adding over all sliding receptive area of the 
                  # input image, we get
                  
                  dx[n1, c1, h1*stride:h1*stride+pool_height, w1*stride:w1*stride+pool_width] += update_dx
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


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

