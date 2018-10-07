import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

# MY helper functions *****************************************************

def my_ReLU(z):
    return z * (z>0)# the ReLU function (element wise multiplication) (NxH)

def my_softmax(scores):
  # For stability purposes, we should first subtract the max, so that the 
  # exponential does not blow up and then apply the softmax, Note: this does 
  # not change the expression, as we are multiplying and dividing by e^{max}
  # to the softmax function(http://cs231n.github.io/linear-classify/#softmax)

  # columnwise max 
  scores = (scores.transpose() -np.max(scores, axis=1)).transpose()
  exp_scores = np.exp(scores)
  sum_scores = np.sum(exp_scores, axis=1)
  return (exp_scores.transpose()/sum_scores).transpose() # NxC


class mult_cell: 
    def __init__(self, m1, m2):# m2 = W2
        self.m1 = m1
        self.m2 = m2
        self.fp = np.dot(self.m1, self.m2)
        self.grad = None
        self.bp = None

    def forward_pass(self):
        return self.fp

    def backward_pass(self, flag):
        if flag==0: # derivative wrt the first term 
            return self.m2
        else:
            return self.m1


class add_cell:
    def __init__(self, a1, a2):
        self.fp = np.add(a1, a2)
        self.bp = 1
        self.grad = None

    def forward_pass(self):
        return self.fp

    def backward_pass(self):
        return self.bp


class ReLU_cell: # element wise ReLU
    def __init__(self, m1):
        self.m1 = m1
        N,H = np.shape(m1)
        self.fp = self.__my_ReLU()
        self.bp = np.sum((self.fp*1>0)*1, axis=0)*1/N # derivative of ReLU, '0' if x==0
#        print(self.bp)
        self.grad = None

    def __my_ReLU(self):
        return self.m1 * (self.m1>0)# the ReLU function (element wise multiplication) (NxH)
    
    def forward_pass(self):
        return self.fp
    
    def backward_pass(self):
        return self.bp


class softmax_cell:
    def __init__(self, m1):
        self.m1 = m1
        self.fp = self.__my_softmax(self.m1)
        self.bp = None
        self.grad = None

    def __my_softmax(self, scores):
        # For stability purposes, we should first subtract the max, so that the 
        # exponential does not blow up and then apply the softmax, Note: this does 
        # not change the expression, as we are multiplying and dividing by e^{max}
        # to the softmax function(http://cs231n.github.io/linear-classify/#softmax)

        # columnwise max
        scores = (scores.transpose() -np.max(scores, axis=1)).transpose()
        exp_scores = np.exp(scores)
        sum_scores = np.sum(exp_scores, axis=1)
        return (exp_scores.transpose()/sum_scores).transpose() # NxC

    def forward_pass(self):
        return self.fp

    def backward_pass(self):
        N,C = np.shape(self.m1)
#        print(np.shape(self.m1))
        self.bp = np.zeros((C, C))
#        print('self.bp', self.bp)
#        print('self.fp', self.fp)
#        print('self.m1', self.m1)
        for i in range(N):
            # Jacobian of self.fp[i] wrt self.m1[i]
            for p in range(C):
                for q in range(C):
                    if p==q:
                        self.bp[p][q] += self.fp[i][p]*(1-self.fp[i][p])
                    else:
                        self.bp[p][q] += -1*self.fp[i][p]*self.fp[i][q]
        return self.bp*1/N


class log_cell:
    def __init__(self, input_m1):
        self.N,_ = np.shape(input_m1) 
        self.input = input_m1
        self.fp = np.log(self.input)
        self.bp = np.sum(1/self.input, axis=0)# will be a Jacobian (diag matrix)
        self.grad = None
    
    def forward_pass(self):# forward pass
        return self.fp

    def backward_pass(self):# differentiate the self.fp wrt input_vector
        return self.bp*1/self.N # multiply this by the input gradient element-wise


class total_loss_cell:
    def __init__(self, y_hot, m1, W1, W2, reg):
        self.y = y_hot
        self.m1 = m1
        N = len(self.m1)
        reg_loss = 0.5*reg*(np.sum(W1**2) + np.sum(W2**2)) # 1/2*lambda(W1^2+W2^2)
        self.fp = -np.sum([np.dot(self.y[i], self.m1[i]) for i in range(N)])*1/N + reg_loss
        self.bp = -np.sum(self.y, axis=0) * 1/N
        self.grad = None

    def forward_pass(self):
        return self.fp

    def backward_pass(self):
        return self.bp

# Helper functions end here************************************************

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization the the weight matrices. The two layer net should use a ReLU
  nonlinearity after the first affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape


  # compute the forward pass
  scores = None
  #############################################################################
  # Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  # say the hidden layer output is h1.
  t2 = np.dot(X, W1) + b1
  t3 = my_ReLU(t2)
#  print('t3', t3)
  t5 = np.dot(t3, W2) + b2 # NxC
  scores = t5
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################
  
  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  y = np.array(y)
  C = len(set(y)) # number of classes

  # compute the loss
  loss = None
  #############################################################################
  # Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss. So that your results match ours, multiply the            #
  # regularization loss by 0.5                                                #
  #############################################################################

  softmax_scores = my_softmax(scores)
#  print(softmax_scores)
  # cross-entropy loss is the data loss for softmax classifier
  data_loss = -np.sum(np.log(softmax_scores[[i for i in range(N)], y]))*1/N
  reg_loss = 0.5*reg*(np.sum(W1**2) + np.sum(W2**2)) # 1/2*lambda(W1^2+W2^2)
  loss = data_loss + reg_loss


  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
#  grads['W1'] = 
  dt5 = softmax_scores
  dt5[range(N), y] -= 1
  dt5 /= N
  
  dt4 = dt5
  dt3 = np.dot(dt4, W2.T)
  dt2 = dt3*((t3>0)*1)
  dt1 = dt2

#  print(dt5)
  grads['b2'] = np.sum(dt5, axis=0)
  grads['W2'] = np.dot(t3.T, dt4) + reg*W2
  grads['b1'] = np.sum(dt2, axis=0)
  grads['W1'] = np.dot(X.T, dt1) + reg*W1
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads

