# Authors: Jonathan Siegel and Andrea Bonito
#
# Tests the consistent formulation of PINNs against the original L2 loss formulation on two elliptic problems.

import math
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import grad, value_and_grad
from jax import jacfwd, jacrev,jit
from utils import plot_values
from experiments import generate_elliptic_experiment
from networks import init_deep_network_params, batched_predict, batched_grad_predict, batched_laplacians_predict

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

### Define the two Loss Functions. Input are the parameters, collocation points, and the data at the collocation points.
# gamma = 1, h12_weight = 1 corresponds to consistent PINNs
# gamma = 2, h12_weight = 0 corresponds to regular PINNS
def loss(params, vals, vals_bdy, rhs_data, bdy_data, gamma, h12_weight):
  lap_vals = batched_laplacians_predict(params, vals)

  # bulk term, total number of terms is N*N
  sum=jnp.absolute(lap_vals+rhs_data.reshape(-1,))
  count = jnp.size(rhs_data)
  output=jnp.power((1.0 / (count))*jnp.sum(jnp.power(sum,jnp.full(count,gamma))),2./gamma)

  # L2 bdy term
  total_no_points = jnp.size(vals_bdy)

  bdy_vals = batched_predict(params,vals_bdy)
  diff = bdy_vals-bdy_data
  bdy = (1.0 / total_no_points)*jnp.sum(jnp.power(diff, 2))
  output += bdy

  # Now add the H^1/2 semi-norm if needed
  if h12_weight > 0:

    cx = vals_bdy[:,0].reshape((jnp.shape(vals_bdy)[0],1))
    cy = vals_bdy[:,1].reshape((jnp.shape(vals_bdy)[0],1))

    # difference matrices
    Mcx = cx.T - cx
    Mcy = cy.T - cy
    M = diff.T - diff

    # strictly upper triangular indices
    id = jnp.triu_indices(len(diff), k=1)

    # denominator |x - y|^2
    norm_diff_sqr = jnp.multiply(Mcx[id],Mcx[id])+jnp.multiply(Mcy[id],Mcy[id])

    bdy += (1.0 / (total_no_points*total_no_points))*jnp.sum(jnp.divide(jnp.power(M[id],jnp.full(M[id].shape[0],2)),norm_diff_sqr))
    output+=h12_weight*bdy

  return output

### Construct the Training Algorithm which is a variant of rescaled gradient descent.
@jit
def update_consistent_loss(params, vals, vals_bdy, rhs_data, bdy_data, velocities, step, mom):
  loss_value, grads = value_and_grad(loss)(params, vals, vals_bdy, rhs_data, bdy_data, 1.1, 1)
  if velocities == None:
    velocities = grads
  else:
    velocities = [(mom * vw + (1-mom) * dw, mom * vb + (1-mom) * db)
                  for (vw, vb), (dw, db) in zip(velocities, grads)]
  return [(w - step * vw / jnp.max(jnp.abs(w)) / (jnp.max(jnp.abs(vw)) + 1e-5), b - step * vb * jnp.max(jnp.abs(b)) / (jnp.max(jnp.abs(vb)) + 1e-5))
          for (w, b), (vw, vb) in zip(params, velocities)], loss_value, velocities

@jit
def update_original_loss(params, vals, vals_bdy, rhs_data, bdy_data, velocities, step, mom):
  loss_value, grads = value_and_grad(loss)(params, vals, vals_bdy, rhs_data, bdy_data, 2, 0)
  if velocities == None:
    velocities = grads
  else:
    velocities = [(mom * vw + (1-mom) * dw, mom * vb + (1-mom) * db)
                  for (vw, vb), (dw, db) in zip(velocities, grads)]
  return [(w - step * vw / jnp.max(jnp.abs(w)) / (jnp.max(jnp.abs(vw)) + 1e-5), b - step * vb * jnp.max(jnp.abs(b)) / (jnp.max(jnp.abs(vb)) + 1e-5))
          for (w, b), (vw, vb) in zip(params, velocities)], loss_value, velocities

def train(params, vals, vals_bdy, rhs_data, bdy_data, step, mom, loss_type, num_steps, step_decrease_int):
  velocities = None
  for epoch in range(num_steps):
    if loss_type == 'original':
      params, loss_value, velocities = update_original_loss(params, vals, vals_bdy, rhs_data, bdy_data, velocities, step, mom)
    else:
      params, loss_value, velocities = update_consistent_loss(params, vals, vals_bdy, rhs_data, bdy_data, velocities, step, mom) 
    if epoch % step_decrease_int == step_decrease_int - 1:
      step = step / 2
  print('Final loss value achieved: %lf' % loss_value)
  return params

def train_and_test(N, Ntest, exp_type, step_size, momentum, loss_type, step_count, decrease_interval, plot = False):
  # Initialize the network randomly.
  params = init_deep_network_params(2, width, depth, random.PRNGKey(0))

  # Generate the data.
  coords, bdy_coords, coords_test, rhs_data, bdy_data, sol, sol_grads = generate_elliptic_experiment(N, Ntest, exp_type)

  # Train the network based upon both the consistent and original loss.
  params = train(params, coords, bdy_coords, rhs_data, bdy_data, step_size, momentum, loss_type, step_count, decrease_interval)

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
