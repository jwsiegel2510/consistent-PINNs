# Author: Jonathan Siegel
#
# Tests consistent PINNs on the 3d Poisson equation.

import sys
import math
import jax.numpy as jnp
from jax import random
from implementation.utils import plot_values
from implementation.experiments import generate_3d_poisson_experiment
from implementation.networks import ResidualReLUkNetwork
from implementation.loss_functions import OriginalPoissonPINNsLoss, ConsistentPoissonPINNsLoss
from implementation.optimization import natural_newton_train

### Tested number of colloation points in each direction and along the boundary.
Nlist = [10]

### Number of points in each direction for plotting and for calculating the error.
Ntest = 100

### Neural Network and training parameters
width = 500
depth = 1
num_steps = 250

def train_and_test(N, Ntest, exp_type, loss_type):
  # Initialize the network randomly.
  network = ResidualReLUkNetwork()
  params = network.init_deep_network_params(3, width, depth, random.PRNGKey(0))

  # Generate the data.
  coords, bdy_coords, coords_test, rhs_data, bdy_data, sol, sol_grads = generate_3d_poisson_experiment(N, Ntest, exp_type)

  # Create loss function.
  if loss_type == 'original':
    loss = OriginalPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data)
  elif loss_type == 'original-weighted':
    loss = OriginalPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data, bdy_weight = N)
  elif loss_type == 'consistent-l2':
    loss = ConsistentPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data, 2.0)
  else:
    # Use a value of gamma = 6/5 in 2d.
    loss = ConsistentPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data, 6.0/5.0)

  # Train the network.
  params = natural_newton_train(params, network, loss, num_steps=num_steps, verbose=True)

  # Calculate and return the relative H1 error.
  nn_sol = network.batched_predict(params, coords_test)

  # Calculate the H1 error.
  nn_grads = network.batched_grad_predict(params, coords_test)

  solution_norm = (1.0/Ntest)*jnp.linalg.norm(sol_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(sol)
  error = (1.0/Ntest)*jnp.linalg.norm(sol_grads - nn_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(nn_sol - sol)

  return error / solution_norm

### Run the experiments.
experiment = 'smooth'
for N in Nlist:
  print('Number of collocation points in each direction: %d' % N)

  error = train_and_test(N, Ntest, experiment, 'original')
  print('Using the original loss gives a relative error of: %lf' % error)

  error = train_and_test(N, Ntest, experiment, 'original-weighted')
  print('Using the weighted original loss gives a relative error of: %lf' % error)

  error = train_and_test(N, Ntest, experiment, 'consistent')
  print('Using the consistent loss gives a relative error of: %lf' % error)

  error = train_and_test(N, Ntest, experiment, 'consistent-l2')
  print('Using the consistent loss with L2 gives a relative error of: %lf' % error)
