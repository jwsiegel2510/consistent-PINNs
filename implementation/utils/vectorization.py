# Author: Jonathan Siegel
#
# Contains methods for vectorizing parameter arrays. This is useful for implementing second order methods.

import jax.numpy as jnp
from jax import jit

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
