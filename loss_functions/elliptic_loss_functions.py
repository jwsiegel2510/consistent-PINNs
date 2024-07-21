# Author: Jonathan Siegel and Andrea Bonito
#
# Contains classes which implement both the original L2 loss function and the consistent loss function for the Poisson equation.

import math
import jax.numpy as jnp

class OriginalPoissonPINNsLoss:
  def __init__(self, coords, bdy_coords, rhs_data, bdy_data):
    self.coords = coords
    self.bdy_coords = bdy_coords
    self.rhs_data = rhs_data
    self.bdy_data = bdy_data

  def apply(self, lap_vals, bdy_vals):
    sum=jnp.absolute(lap_vals+self.rhs_data.reshape(-1,))
    count = jnp.size(self.rhs_data)
    domain_term = (1.0 / (count)) * jnp.sum(jnp.power(sum,jnp.full(count,2.0)))

    diff = bdy_vals - self.bdy_data
    bdy_term = (1.0 / jnp.size(diff)) * jnp.sum(jnp.power(diff, 2))

    return domain_term + bdy_term

class ConsistentPoissonPINNsLoss:
  def __init__(self, coords, bdy_coords, rhs_data, bdy_data, gamma):
    self.coords = coords
    self.bdy_coords = bdy_coords
    self.rhs_data = rhs_data
    self.bdy_data = bdy_data
    self.gamma = gamma

  def apply(self, lap_vals, bdy_vals):
    sum=jnp.absolute(lap_vals+self.rhs_data.reshape(-1,))
    count = jnp.size(self.rhs_data)
    domain_term = jnp.power((1.0 / (count))*jnp.sum(jnp.power(sum,jnp.full(count,self.gamma))),2./self.gamma)

    diff = bdy_vals - self.bdy_data
    num_bdy_points = jnp.size(diff)
    bdy_term = (1.0 / num_bdy_points) * jnp.sum(jnp.power(diff, 2))

    cx = self.bdy_coords[:,0].reshape((jnp.shape(self.bdy_coords)[0],1))
    cy = self.bdy_coords[:,1].reshape((jnp.shape(self.bdy_coords)[0],1))

    # difference matrices
    Mcx = cx.T - cx
    Mcy = cy.T - cy
    M = diff.T - diff

    # strictly upper triangular indices
    id = jnp.triu_indices(len(diff), k=1)

    # denominator |x - y|^2
    norm_diff_sqr = jnp.multiply(Mcx[id],Mcx[id]) + jnp.multiply(Mcy[id],Mcy[id])

    bdy_term += (1.0 / (num_bdy_points * num_bdy_points)) * jnp.sum(jnp.divide(jnp.power(M[id],jnp.full(M[id].shape[0],2)),norm_diff_sqr))

    return domain_term + bdy_term
