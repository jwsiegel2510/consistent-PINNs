# Author: Jonathan Siegel
#
# Test the optimization of both the original and consistent PINNs formulation using a natural gradient Newton's method.
# This results in significantly faster optimization with smaller networks for problems with a smooth solution.

import math
import jax.numpy as jnp
from jax import random
from utils import plot_values
from experiments import generate_elliptic_experiment
from networks import ResidualReLUkNetwork
from loss_functions import OriginalPoissonPINNsLoss, ConsistentPoissonPINNsLoss
from optimization import natural_newton_train

### Tested number of colloation points in each direction and along the boundary.
Nlist = [5, 10, 15, 20, 25, 30]

### Number of points in each direction for plotting and for calculating the error.
Ntest = 500

### Neural Network and training parameters
width = 10
depth = 4

def train_and_test(N, Ntest, exp_type, loss_type, plot = True):
  # Initialize the network randomly.
  network = ResidualReLUkNetwork()
  params = network.init_deep_network_params(2, width, depth, random.PRNGKey(0))

  # Generate the data.
  coords, bdy_coords, coords_test, rhs_data, bdy_data, sol, sol_grads = generate_elliptic_experiment(N, Ntest, exp_type)

  # Create loss function.
  if loss_type == 'original':
    loss = OriginalPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data)
  elif loss_type == 'original-weighted':
    loss = OriginalPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data, bdy_weight = 10.0)
  else:
    # Use a value of gamma = 1.1.
    loss = ConsistentPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data, 1.1)

  # Train the network.
  params = natural_newton_train(params, network, loss)

  # Calculate and return the relative H1 error.
  nn_sol = network.batched_predict(params, coords_test)
  xp_test=jnp.linspace(0.,1.,Ntest)
  yp_test=jnp.linspace(0.,1.,Ntest)

  X_test, Y_test = jnp.meshgrid(xp_test, yp_test)
  if plot:
    plot_values(X_test,Y_test,jnp.reshape(nn_sol,jnp.shape(X_test)))
    plot_values(X_test,Y_test,sol)

  # Calculate the H1 error.
  nn_grads = network.batched_grad_predict(params, coords_test)

  solution_norm = (1.0/Ntest)*jnp.linalg.norm(sol_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(sol)
  error = (1.0/Ntest)*jnp.linalg.norm(sol_grads - nn_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(jnp.reshape(nn_sol, jnp.shape(X_test)) - sol)

  return error / solution_norm

### Run the experiments.
for N in Nlist:
  print('Number of collocation points in each direction: %d' % N)
  
  error = train_and_test(N, Ntest, 'smooth', 'original')
  print('Using the original loss function gives a relative error of: %lf' % error)

  error = train_and_test(N, Ntest, 'smooth', 'original-weighted')
  print('Using the weighted original loss function gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, 'smooth', 'consistent')
  print('Using the consistent loss function gives a relative error of: %lf' % error)
