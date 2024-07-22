# Author: Jonathan Siegel
#
# Contains an implementation of the natural gradient descent method with momentum for training neural networks. 
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

def evaluate_output(vec_params, signature, network, loss):
  """Evaluates the network and its laplacian output on the set of training points.

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
  return jnp.concatenate((lap_vals.flatten(), bdy_vals.flatten()))

def update(params, velocities, network, loss, step, regularization, mom):
  """ Performs one step of Gauss-Newton iteration.
  Args:
    params: Initial network parameters
    velocitiesL velocities from the previous step
    network: Class containing the network evaluation function
    loss: class containing the loss function
    step: stepsize
    regularization: multiple of the identity which is added to the Hessian

  Returns:
    params: new parameter values
    loss_value: current value of the loss function
  """
  vec_list = vectorize(params)
  vec_params = vec_list[0]
  signature = vec_list[1]
  loss_value, grads = value_and_grad(evaluate_loss)(vec_params, signature, network, loss)
  jacobian = jacfwd(evaluate_output)(vec_params, signature, network, loss)
  direction = jnp.linalg.solve(regularization * jnp.identity(grads.size) + jnp.matmul(jnp.transpose(jacobian), jacobian), grads)
  if velocities is None:
    velocities = direction
  else:
    velocities = mom * velocities + (1-mom) * direction
  vec_params -= step * velocities
  return restore(vec_params, signature), velocities, loss_value

def gauss_newton_train(params, network, loss, step = 1.0, regularization = 0.01, num_steps=500, mom = 0.0, verbose = True):
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
  velocities = None
  for epoch in range(num_steps):
    params, velocities, loss_value = update(params, velocities, network, loss, step, regularization, mom)
    if verbose:
      print('epoch: '+ str(epoch)+'   loss value: '+str(loss_value))
  return params

