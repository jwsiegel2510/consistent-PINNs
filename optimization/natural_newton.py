# Author: Jonathan Siegel
#
# Contains an implementation of the natural gradient Newton's method for training neural networks. 
# The method is only suitable for training small networks due to its complexity in terms of the number of parameters.

import math
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, jacfwd
from functools import partial

def vectorize(params):
  """Recursive function for vectorizing parameters.

  Args:
    params: list of parameters

  Returns:
    vec_params: vector of parameters
    signature: list of tensor shapes for un-vectorizing parameters
  """
  if type(params) is list:
    unpack_list = [vectorize(p) for p in params]
    vec_params = jnp.concatenate([ul[0] for ul in unpack_list])
    signature = [ul[1] for ul in unpack_list]
    return [vec_params, signature]
  else:
    signature = params.shape
    return [params.flatten(), signature]

def restore(vec_params, signature):
  params, end = restore_rec(vec_params, signature, 0)
  return params

def restore_rec(vec_params, signature, start):
  """Recursively undoes the parameter vectorization.

  Args:
    vec_params: vector of parameters
    signature: list of tensor shapes for un-vectorizing parameters
    start: index to start at

  Returns:
    params: original list of parameters
    start: new index to start at
  """
  if type(signature) is list:
    params = []
    for sigs in signature:
      temp_params, start = restore_rec(vec_params, sigs, start)
      params.append(temp_params)
    return params, start
# Recursive function for vectorizing parameters.
  else:
    # In this case signature is a tuple.
    length = jnp.prod(jnp.array(signature))
    return jnp.reshape(vec_params[start:start+length], signature), start+length

def evaluate_loss(vec_params, signature, network, loss):
  """Evaluates the loss function on a vectorized set of parameters.

  Args:
    vec_params: vector of params
    signature: nested list of tuples giving how to unpack the parameters
    network: neural network class
    loss: loss function class

  Returns:
    Value of the loss function
  """
  params = restore(vec_params, signature)
  
  # Extract sample point coordinates from loss and evaluate network at sample points
  coords = loss.coords
  bdy_coords = loss.bdy_coords
  lap_vals = network.batched_laplacians_predict(params, coords)
  bdy_vals = network.batched_predict(params, bdy_coords)
 
  # Evaluate loss
  return loss.apply(lap_vals, bdy_vals)

def evaluate_laps(vec_params, signature, network, loss):
  """Evaluates the network and its laplacian output on the set of training points.

  Args:
    vec_params: vector of params
    signature: nested list of tuples giving how to unpack the parameters
    network: neural network class
    loss: loss function class

  Returns:
    Values of the network laplacian.
  """
  params = restore(vec_params, signature)
  
  # Extract sample point coordinates from loss and evaluate network at sample points
  coords = loss.coords
  lap_vals = network.batched_laplacians_predict(params, coords)
  return lap_vals.flatten()

def evaluate_bdy(vec_params, signature, network, loss):
  """Evaluates the network and its laplacian output on the set of training points.

  Args:
    vec_params: vector of params
    signature: nested list of tuples giving how to unpack the parameters
    network: neural network class
    loss: loss function class

  Returns:
    boundary_values of the network
  """
  params = restore(vec_params, signature)
  
  # Extract sample point coordinates from loss and evaluate network at sample points
  bdy_coords = loss.bdy_coords
  bdy_vals = network.batched_predict(params, bdy_coords)
  return bdy_vals.flatten()


def update(params, network, loss, regularization):
  """ Performs one step of Gauss-Newton iteration.
  Args:
    params: Initial network parameters
    velocitiesL velocities from the previous step
    network: Class containing the network evaluation function
    loss: class containing the loss function
    regularization: multiple of the identity which is added to the Hessian

  Returns:
    params: new parameter values
    loss_value: current value of the loss function
  """
  vec_list = vectorize(params)
  vec_params = vec_list[0]
  signature = vec_list[1]
  loss_value, grads = value_and_grad(evaluate_loss)(vec_params, signature, network, loss)
  jacobian_laps = jacfwd(evaluate_laps)(vec_params, signature, network, loss)
  laps_gram_matrix = jnp.matmul(jnp.matmul(jnp.transpose(jacobian_laps), loss.domain_mat), jacobian_laps)
  jacobian_bdy = jacfwd(evaluate_bdy)(vec_params, signature, network, loss)
  bdy_gram_matrix = jnp.matmul(jnp.matmul(jnp.transpose(jacobian_bdy), loss.bdy_mat), jacobian_bdy)
  direction = jnp.linalg.solve(regularization * jnp.identity(grads.size) + laps_gram_matrix + bdy_gram_matrix, grads)
  # Implement a line search to find a good step size.
  steps = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
  leave = True
  for step in steps:
    vec_params_test = vec_params - step * direction
    loss_value_test, grads = value_and_grad(evaluate_loss)(vec_params_test, signature, network, loss)
    if loss_value_test < loss_value:
      vec_params = vec_params_test
      loss_value = loss_value_test
      leave = False
  return restore(vec_params, signature), loss_value, leave

def natural_newton_train(params, network, loss, regularization = 0.01, max_num_steps=500, verbose = True):
  """Train the neural network on the given loss function using the Gauss-Newton method with the given hyperparameters.

  Args:
    params: Initial network parameters
    network: Class containing the network evaluation function
    loss: class containing the loss function
    step: stepsize
    regularization: parameter for regularizing Newton step
    num_steps: number of training steps
    verbose: Indicates if detailed training information should be printed

  Returns:
    New value of the parameters
  """
  for epoch in range(max_num_steps):
    params, loss_value, leave = update(params, network, loss, regularization)
    if verbose:
      print('epoch: '+ str(epoch)+'   loss value: '+str(loss_value))
    if leave:
      return params
  return params

