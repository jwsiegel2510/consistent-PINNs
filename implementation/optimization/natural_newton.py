# Author: Jonathan Siegel
#
# Contains an implementation of the natural gradient Newton's method for training neural networks. 
# The method is only suitable for training small networks due to its complexity in terms of the number of parameters.

import math
import jax.numpy as jnp
from jax import jit
from functools import partial
from ..utils import vectorize, restore

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
  # Calculate gradient of the loss function
  grads = loss.gradient(params, network)

  # Vectorize both the gradients and parameters
  vec_list = vectorize(grads)
  vec_grads = vec_list[0]
  signature = vec_list[1]
  vec_params = vectorize(params)[0]
  
  # Obtain the natural gradient quadratic form from the loss function
  natural_grad_form = loss.natural_gradient_form(params, network)

  # Determine search direction by solving the linear system
  direction = jnp.linalg.solve(regularization * jnp.identity(vec_grads.size) + natural_grad_form, vec_grads)

  # Implement a line search to find a good step size.
  loss_value = 0.0
  steps = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  for step in steps:
    vec_params_test = vec_params - step * direction
    loss_value_test = loss.evaluate(restore(vec_params_test, signature), network)
    if step == steps[0] or loss_value_test < loss_value:
      vec_params = vec_params_test
      loss_value = loss_value_test
  return restore(vec_params, signature), loss_value

def natural_newton_train(params, network, loss, regularization = 0.01, num_steps=250, verbose = True):
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
  for epoch in range(num_steps):
    params, loss_value = update(params, network, loss, regularization)
    if verbose:
      print('epoch: '+ str(epoch)+'   loss value: '+str(loss_value))
  print('total epoch: '+ str(num_steps)+'   final loss value: '+str(loss_value))
  return params

