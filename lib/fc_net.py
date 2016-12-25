import numpy as np

from layers import *
from layer_utils import *

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

  def __init__(self, hidden_dims, input_dim, num_classes,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: std for weight initialization
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    assert type(hidden_dims) is list, '"hidden_dims" must be a list of numbers'
    self.N = input_dim
    self.C = num_classes
    dims = [self.N] + hidden_dims + [self.C]
    Ws = {'W' + str(i+1): np.random.randn(dims[i], dims[i+1]) * weight_scale 
          for i in xrange(self.num_layers)}
    bs = {'b' + str(i+1): np.zeros(dims[i+1]) for i in xrange(self.num_layers)}
    self.params.update(Ws)
    self.params.update(bs)

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
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
      gammas = {'gamma' + str(i+1): np.ones(dims[i+1])
                for i in xrange(self.num_layers - 1)}
      betas = {'beta' + str(i+1): np.zeros(dims[i+1])
                for i in xrange(self.num_layers - 1)}
      self.params.update(gammas)
      self.params.update(betas)
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

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
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    caches = {} # store all caches during forward pass
    out = X # output of previous layer
    # forward prop from the first layer to the second to last layer
    for i in xrange(1, self.num_layers):
      tag = str(i)
      if self.use_batchnorm:
        # affine--bn--relu
        out, caches[i] = affine_bn_relu_forward(out, self.params['W' + tag],
          self.params['b' + tag], self.params['gamma' + tag], 
          self.params['beta' + tag], self.bn_params[i-1])
      else:
        # affine--relu
        out, caches[i] = affine_relu_forward(out, self.params['W' + tag], 
          self.params['b' + tag])
      if self.use_dropout:
        # use dropout after each relu
        out, caches['drop' + tag] = dropout_forward(out, self.dropout_param)
    # forward prop to the last layer, affine
    out, caches[self.num_layers] = affine_forward(out, 
      self.params['W' + str(self.num_layers)],
      self.params['b' + str(self.num_layers)])

    # If test mode return early
    if mode == 'test':
      return out

    grads = {}

    loss, dout = softmax_loss(out, y)
    # last layer, affine backward
    dout, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)
      ] = affine_backward(dout, caches[self.num_layers])
    # add L2 regularization terms to loss and gradients
    loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)]**2)
    grads['W' + str(self.num_layers)] += self.reg * self.params['W' + 
                                          str(self.num_layers)]
    # from second to last layer backprop to the first layer
    for i in xrange(self.num_layers - 1, 0, -1):
      tag = str(i)
      # use dropout
      if self.use_dropout:
        dout = dropout_backward(dout, caches['drop' + tag])
      if self.use_batchnorm:
        # affine--bn--relu
        dout, dW, db, dgamma, dbeta = affine_bn_relu_backward(dout, caches[i])
        grads['W' + tag], grads['b' + tag] = dW, db
        grads['gamma' + tag], grads['beta' + tag] = dgamma, dbeta
      else:
        # affine--relu
        dout, grads['W' + tag], grads['b' + tag] = affine_relu_backward(
          dout, caches[i])
      # add L2 regularization terms to loss and gradients
      loss += 0.5 * self.reg * np.sum(self.params['W' + tag] ** 2)
      grads['W' + tag] += self.reg * self.params['W' + tag]

    return loss, grads
