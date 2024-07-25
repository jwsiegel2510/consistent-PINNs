# Author: Jonathan Siegel and Andrea Bonito
#
# Contains classes which implement both the original L2 loss function and the consistent loss function for the Poisson equation.

import math
import jax.numpy as jnp

# Ad-hoc function for reproducing some of numpys features.
def fill_diagonal(a, val):
  assert a.ndim >= 2
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  return a.at[..., i, j].set(val)

class OriginalPoissonPINNsLoss:
  def __init__(self, coords, bdy_coords, rhs_data, bdy_data):
    self.coords = coords
    self.bdy_coords = bdy_coords
    self.rhs_data = rhs_data
    self.bdy_data = bdy_data

    # Construct loss matrices
    domain_size = jnp.size(self.rhs_data)
    self.domain_mat = (1.0 / domain_size) * jnp.identity(domain_size)
    bdy_size = jnp.size(self.bdy_data)
    self.bdy_mat = (1.0 / (2.0 * bdy_size)) * jnp.identity(bdy_size)

  def apply(self, lap_vals, bdy_vals):
    diff = lap_vals + self.rhs_data.reshape(-1,)
    domain_term = jnp.matmul(jnp.matmul(diff.transpose(), self.domain_mat), diff)

    diff = bdy_vals.reshape(-1,) - self.bdy_data.reshape(-1,)
    bdy_term = jnp.matmul(jnp.matmul(diff.transpose(), self.bdy_mat), diff)

    return domain_term + bdy_term

class ConsistentPoissonPINNsLoss:
  def __init__(self, coords, bdy_coords, rhs_data, bdy_data, gamma):
    self.coords = coords
    self.bdy_coords = bdy_coords
    self.rhs_data = rhs_data
    self.bdy_data = bdy_data
    self.gamma = gamma

    # Construct loss matrices
    domain_size = jnp.size(self.rhs_data)
    self.domain_mat = (1.0 / domain_size) * jnp.identity(domain_size)

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
    self.bdy_mat = 0.5 * (1.0 / (bdy_size * bdy_size)) * fill_diagonal(-1.0 * norm_diff_sqr, new_diag) + (1.0 / (2.0 * bdy_size)) * jnp.identity(bdy_size)

  def apply(self, lap_vals, bdy_vals):
    diff = jnp.power(jnp.abs(lap_vals + self.rhs_data.reshape(-1,)) + 1e-8, self.gamma / 2.0)
    domain_term = jnp.power(jnp.matmul(jnp.matmul(diff.transpose(), self.domain_mat), diff), 2.0 / self.gamma)

    diff = bdy_vals.reshape(-1,) - self.bdy_data.reshape(-1,)
    bdy_term = jnp.matmul(jnp.matmul(diff.transpose(), self.bdy_mat), diff)

    return domain_term + bdy_term

