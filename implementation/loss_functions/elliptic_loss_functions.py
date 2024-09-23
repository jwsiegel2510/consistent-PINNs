# Author: Jonathan Siegel and Andrea Bonito
#
# Contains classes which implement both the original L2 loss function, the consistent loss function, and the deep Ritz loss with consistent boundary terms for the 2d Poisson equation.

import math
import jax.numpy as jnp
from jax import grad, jacfwd, jit
from functools import partial
from ..utils import vectorize, restore

# Ad-hoc function for reproducing some of numpys features.
def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)

class OriginalPoissonPINNsLoss:
  def __init__(self, coords, bdy_coords, rhs_data, bdy_data, bdy_weight = 1.0):
    self.coords = coords
    self.bdy_coords = bdy_coords
    self.rhs_data = rhs_data
    self.bdy_data = bdy_data

    # Construct loss matrices
    domain_size = jnp.size(self.rhs_data)
    self.d_mat = (1.0 / domain_size) * jnp.identity(domain_size)
    bdy_size = jnp.size(self.bdy_data)
    self.b_mat = (bdy_weight / (2.0 * bdy_size)) * jnp.identity(bdy_size)

  #@partial(jit, static_argnums=[0,2])
  def evaluate(self, params, network):
    lap_vals = network.batched_laplacians_predict(params, self.coords)
    diff = lap_vals + self.rhs_data
    domain_term = jnp.matmul(jnp.matmul(diff.transpose(), self.d_mat), diff)

    bdy_vals = network.batched_predict(params, self.bdy_coords)
    diff = bdy_vals - self.bdy_data
    bdy_term = jnp.matmul(jnp.matmul(diff.transpose(), self.b_mat), diff)

    return domain_term + bdy_term

  @partial(jit, static_argnums=[0,2])
  def gradient(self, params, network):
    loss = lambda parameters: self.evaluate(parameters, network)
    return grad(loss)(params)

  def natural_gradient_form(self, params, network):
    # Vectorize parameters
    vec_list = vectorize(params)
    vec_params = vec_list[0]
    signature = vec_list[1]
 
    # Calculate jacobian matrices for laplacian values and boundary values
    evaluate_laps = lambda parameters: network.batched_laplacians_predict(restore(parameters, signature), self.coords)
    jacobian_laps = jacfwd(evaluate_laps)(vec_params)
    evaluate_bdy = lambda parameters: network.batched_predict(restore(parameters, signature), self.bdy_coords)
    jacobian_bdy = jacfwd(evaluate_bdy)(vec_params)

    # Return the natural gradient quadratic form
    laps_gram_matrix = jnp.matmul(jnp.matmul(jnp.transpose(jacobian_laps), self.d_mat), jacobian_laps)
    bdy_gram_matrix = jnp.matmul(jnp.matmul(jnp.transpose(jacobian_bdy), self.b_mat), jacobian_bdy)
    return laps_gram_matrix + bdy_gram_matrix

class ConsistentPoissonPINNsLoss:
  def __init__(self, coords, bdy_coords, rhs_data, bdy_data, gamma):
    self.coords = coords
    self.bdy_coords = bdy_coords
    self.rhs_data = rhs_data
    self.bdy_data = bdy_data
    self.gamma = gamma

    # Construct loss matrices
    domain_size = jnp.size(self.rhs_data)
    self.d_mat = (1.0 / domain_size) * jnp.identity(domain_size)

    bdy_size = jnp.size(self.bdy_data)
    cx = self.bdy_coords[:,0].reshape((jnp.shape(self.bdy_coords)[0],1))
    cy = self.bdy_coords[:,1].reshape((jnp.shape(self.bdy_coords)[0],1))

    # difference matrices
    Mcx = cx.T - cx
    Mcy = cy.T - cy
    norm_diff_sqr = jnp.multiply(Mcx,Mcx) + jnp.multiply(Mcy,Mcy)
    norm_diff_sqr = 1.0 / norm_diff_sqr
    norm_diff_sqr = fill_diagonal(norm_diff_sqr, 0)
    new_diag = jnp.sum(norm_diff_sqr, 1)
    self.b_mat = 0.5 * (1.0 / (bdy_size * bdy_size)) * fill_diagonal(-1.0 * norm_diff_sqr, new_diag) + (1.0 / (2.0 * bdy_size)) * jnp.identity(bdy_size)

  @partial(jit, static_argnums=[0,2])
  def evaluate(self, params, network):
    lap_vals = network.batched_laplacians_predict(params, self.coords)
    diff = jnp.power(jnp.abs(lap_vals + self.rhs_data) + 1e-8, self.gamma / 2.0)
    domain_term = jnp.power(jnp.matmul(jnp.matmul(diff.transpose(), self.d_mat), diff), 2.0 / self.gamma)

    bdy_vals = network.batched_predict(params, self.bdy_coords)
    diff = bdy_vals - self.bdy_data
    bdy_term = jnp.matmul(jnp.matmul(diff.transpose(), self.b_mat), diff)

    return domain_term + bdy_term

  @partial(jit, static_argnums=[0,2])
  def gradient(self, params, network):
    loss = lambda parameters: self.evaluate(parameters, network)
    return grad(loss)(params)

  def natural_gradient_form(self, params, network):
    # Vectorize parameters
    vec_list = vectorize(params)
    vec_params = vec_list[0]
    signature = vec_list[1]

    # Calculate jacobian matrices for laplacian values and boundary values
    evaluate_laps = lambda parameters: network.batched_laplacians_predict(restore(parameters, signature), self.coords)
    jacobian_laps = jacfwd(evaluate_laps)(vec_params)
    evaluate_bdy = lambda parameters: network.batched_predict(restore(parameters, signature), self.bdy_coords)
    jacobian_bdy = jacfwd(evaluate_bdy)(vec_params)

    # Return the natural gradient quadratic form
    domain_matrix = self.domain_mat(evaluate_laps(vec_params))
    laps_gram_matrix = jnp.matmul(jnp.matmul(jnp.transpose(jacobian_laps), domain_matrix), jacobian_laps)
    bdy_gram_matrix = jnp.matmul(jnp.matmul(jnp.transpose(jacobian_bdy), self.b_mat), jacobian_bdy)
    return laps_gram_matrix + bdy_gram_matrix

  # The domain matrix must change depending upon the input if gamma != 2.
  @partial(jit, static_argnums=[0])
  def domain_mat(self, lap_vals):
    diff = jnp.power(jnp.abs(lap_vals + self.rhs_data) + 1e-8, self.gamma / 2.0)
    domain_term = jnp.power(jnp.matmul(jnp.matmul(diff.transpose(), self.d_mat), diff), 2.0 / self.gamma)
    return jnp.power(domain_term, ((2.0 - self.gamma) / 2.0)) * (jnp.multiply(self.d_mat, jnp.power(diff, 2.0 * (self.gamma - 2.0) / self.gamma)))

