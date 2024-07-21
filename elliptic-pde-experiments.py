# Authors: Jonathan Siegel and Andrea Bonito
#
# Tests the consistent formulation of PINNs against the original L2 loss formulation on two elliptic problems.

import math
import jax.numpy as jnp
from jax import random
from utils import plot_values
from experiments import generate_elliptic_experiment
from networks import ResidualReLUkNetwork
from loss_functions import OriginalPoissonPINNsLoss, ConsistentPoissonPINNsLoss
from optimization import rgd_train

### Tested number of colloation points in each direction and along the boundary.
Nlist = [5, 10, 15, 20, 25, 30]

### Number of points in each direction for plotting and for calculating the error.
Ntest = 500

### Neural Network and training parameters
width = 100
depth = 8
step_size = 0.001
momentum = 0.9
decrease_interval = 4000
step_count = 40000

def train_and_test(N, Ntest, exp_type, step_size, momentum, loss_type, step_count, decrease_interval, plot = False):
  # Initialize the network randomly.
  network = ResidualReLUkNetwork()
  params = network.init_deep_network_params(2, width, depth, random.PRNGKey(0))

  # Generate the data.
  coords, bdy_coords, coords_test, rhs_data, bdy_data, sol, sol_grads = generate_elliptic_experiment(N, Ntest, exp_type)

  # Create loss function.
  if loss_type == 'original':
    loss = OriginalPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data)
  else:
    loss = ConsistentPoissonPINNsLoss(coords, bdy_coords, rhs_data, bdy_data)

  # Train the network.
  params = rgd_train(params, network, loss, step_size, momentum, step_count, decrease_interval)

  # Calculate and return the relative H1 error.
  nn_sol = batched_predict(params, coords_test)
  xp_test=jnp.linspace(0.,1.,Ntest)
  yp_test=jnp.linspace(0.,1.,Ntest)

  X_test, Y_test = jnp.meshgrid(xp_test, yp_test)
  if plot:
    plot_values(X_test,Y_test,jnp.reshape(nn_solution,jnp.shape(X_test)))
    plot_values(X_test,Y_test,exact_solution)

  # Calculate the H1 error.
  nn_grads = batched_grad_predict(params, coords_test)

  solution_norm = (1.0/Ntest)*jnp.linalg.norm(sol_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(sol)
  error = (1.0/Ntest)*jnp.linalg.norm(sol_grads - nn_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(jnp.reshape(nn_sol, jnp.shape(X_test)) - sol)

  return error / solution_norm

### Run the experiments.
for N in Nlist:
  print('Number of collocation points in each direction: %d' % N)
  
  error = train_and_test(N, Ntest, 'harmonic', step_size, momentum, 'original', step_count, decrease_interval) 
  print('Using the original loss function for the harmonic u gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, 'harmonic', step_size, momentum, 'consistent', step_count, decrease_interval) 
  print('Using the consistent loss function for the harmonic u gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, 'nonsmooth', step_size, momentum, 'original', step_count, decrease_interval) 
  print('Using the original loss function for the nonsmooth u gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, 'nonsmooth', step_size, momentum, 'consistent', step_count, decrease_interval) 
  print('Using the consistent loss function for the nonsmooth u gives a relative error of: %lf' % error)
