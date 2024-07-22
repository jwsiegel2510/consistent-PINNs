# Author: Jonathan Siegel
#
# Contains a neural network training algorithm which is a ad-hoc variant of rescaled gradient descent.

import math
import jax.numpy as jnp
from jax import grad, value_and_grad, jit
from functools import partial

@partial(jit, static_argnums=[1,2])
def evaluate_loss(params, network, loss):
  """Evaluates the loss. We take the gradient of this function.

  Args:
    params: Initial network parameters
    network: Class containing the network evaluation function
    loss: class containing the loss function

  Returns:
    Loss value
  """
  # Extract sample point coordinates from loss and evaluate network at sample points
  coords = loss.coords
  bdy_coords = loss.bdy_coords
  lap_vals = network.batched_laplacians_predict(params, coords)
  bdy_vals = network.batched_predict(params, bdy_coords)
  
  # Evaluate loss
  return loss.apply(lap_vals, bdy_vals)

@partial(jit, static_argnums=[2,3])
def update(params, velocities, network, loss, step, mom):
  """Performs one update step.
  
  Args:
    params: Initial network parameters
    network: Class containing the network evaluation function
    loss: class containing the loss function
    step: stepsize
    mom: momentum parameter

  Returns:
    params: new parameter values
    loss_value: current value of the loss function
    velocities: updated velocities
  """
  loss_value, grads = value_and_grad(evaluate_loss)(params, network, loss)
  if velocities == None:
    velocities = grads
  else:
    velocities = [(mom * vw + (1-mom) * dw, mom * vb + (1-mom) * db)
                  for (vw, vb), (dw, db) in zip(velocities, grads)]
  return [(w - step * vw / jnp.max(jnp.abs(w)) / (jnp.max(jnp.abs(vw)) + 1e-5), b - step * vb * jnp.max(jnp.abs(b)) / (jnp.max(jnp.abs(vb)) + 1e-5))
          for (w, b), (vw, vb) in zip(params, velocities)], loss_value, velocities

def rgd_train(params, network, loss, step, mom, num_steps, step_decrease_int, verbose = False):
  """Train the neural network on the given loss function with the given hyperparameters.

  Args:
    params: Initial network parameters
    network: Class containing the network evaluation function
    loss: class containing the loss function
    step: stepsize
    mom: momentum parameter
    num_steps: number of training steps
    step_decrease_int: frequency with which step size is halved

  Returns:
    New value of the parameters
  """
  velocities = None
  for epoch in range(num_steps):
    params, loss_value, velocities = update(params, velocities, network, loss, step, mom)
    if epoch % step_decrease_int == step_decrease_int - 1:
      step / 2
    if verbose:
      print('epoch: '+ str(epoch)+'   loss value: '+str(loss_value))
  return params
