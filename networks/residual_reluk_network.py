# Author: Jonathan Siegel and Andrea Bonito
#
# Constructs the residual reluk networks used in our tests on elliptic problems.

import jax.numpy as jnp
from jax import random, vmap
from jax import grad, value_and_grad

def random_layer_params(m, n, key, scale=1e-2):
  """A helper function to randomly initialize weights and biases.

  Args:
    m: Input dimension
    n: Output dimension
    key: random key

  Outputs:
    weights and biases of the layer.
  """
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_deep_network_params(input_dim, n, d, key):
  """Initialize all layers for a deep neural network with width n, depth d, and the given input dimension.
  
  Args:
    input_dim: Dimension of the network input
    n: width
    d: number of hidden layers
    key: Random seed
  
  Outputs:
    list of all of the network parameters for each layer
  """
  keys = random.split(key, d+1)
  sizes = [input_dim]
  for i in range(d):
    sizes.append(n)
  sizes.append(1)
  weights = []
  for m, n, k in zip(sizes[:-1], sizes[1:], keys):
    if m == n:
      weights.append(random_layer_params(m, n, k, jnp.sqrt(2) * jnp.power(2.0/15.0, 1.0/6.0) / jnp.sqrt(m * d))) # For ReLU3
    elif n == 1:
      weights.append(random_layer_params(m, n, k, 1.0 / jnp.sqrt(m)))
    else:
      weights.append(random_layer_params(m, n, k, 1.0))
  return weights

# Activation functions
def tanh(x):
  return (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))

def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def relu3(x):
  return jnp.maximum(0,jnp.power(x,3))

def predict(params, values):
  """Calculate output based upon network parameters

  Args:
    params: list of parameters
    values: input to the network

  Output:
    output of the network.
  """
  activations = values
  init_w, init_b = params[0]
  outputs = jnp.dot(init_w,activations)
  outputs = outputs + init_b.reshape(outputs.shape)
  activations = tanh(outputs)
  for w, b in params[1:-1]:
    outputs = jnp.dot(w, activations)
    outputs = outputs + b.reshape(outputs.shape)
    activations = activations + relu3(outputs)

  final_w, final_b = params[-1]
  return (jnp.dot(final_w, activations) + final_b)

# Cast the output to a scalar.
def scalar_predict(params, values):
  return predict(params, values)[0]

# Calculate gradient of input
def grad_predict(params, values):
  return grad(scalar_predict, 1)(params, values)

# Calculate the Laplacian of the neural network function
def predict_laplacians(params, values):
  dxx = grad(lambda params, values: grad_predict(params, values)[0], 1)(params, values)[0]
  dyy = grad(lambda params, values: grad_predict(params, values)[1], 1)(params, values)[1]
  return dxx + dyy

# Enable evaluation at multiple points simultaneously
batched_predict = vmap(predict, in_axes=(None, 0))

# Enable evaluation of gradient at multiple points.
batched_grad_predict = vmap(grad_predict, in_axes=(None, 0))

# Enable evaluation of laplacian at multiple points simultaneously
batched_laplacians_predict = vmap(predict_laplacians, in_axes=(None, 0))
