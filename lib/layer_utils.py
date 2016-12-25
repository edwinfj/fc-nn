from layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenient layer that performs affine--batch norm--relu

  Inputs
  - x: Input to the affine layer
  - w, b: Weights for the affine layer
  - gamma, beta, bn_param: parames for the bn layer
  Outputs
  - out, (fc_cache, bn_cache, relu_cache)
  """
  out, fc_cache = affine_forward(x, w, b)
  out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
  out, relu_cache = relu_forward(out)

  return out, (fc_cache, bn_cache, relu_cache)

def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the affine--bn--relu convenience layer

  INPUT
  - dout, cache
  OUTPUT
  - dout, dw, db, dgamma, dbeta
  """
  fc_cache, bn_cache, relu_cache = cache
  dout = relu_backward(dout, relu_cache)
  dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
  dout, dw, db = affine_backward(dout, fc_cache)

  return dout, dw, db, dgamma, dbeta