# Authors: Jonathan Siegel and Andrea Bonito
#
# Tests the consistent formulation of PINNs against the original L2 loss formulation on two elliptic problems.

import math
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import grad, value_and_grad
from jax import jacfwd, jacrev,jit
import jax.numpy.linalg as jla
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy
from sympy import lambdify

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

### Function for Plotting the Solution
def plot_values(Xval,Yval,solution):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  surf = ax.plot_surface(Xval, Yval, solution, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  plt.show()

### Data Generation for both elliptic problems
x_sym = sympy.Symbol('x')
y_sym = sympy.Symbol('y')
r_sym = sympy.Symbol('r', nonnegative=True)

# Harmonic function tested
u_harmonic = sympy.exp(x_sym)*sympy.cos(sympy.pi*y_sym)
u_harmonic_call = lambdify((x_sym, y_sym), u_harmonic)

# Non-smooth function with vanishing boundary values tested.
u_tmp = 1000*x_sym*(1-x_sym)*y_sym*(1-y_sym)*r_sym**(4.5)
u_nonsmooth = u_tmp.subs({r_sym:sympy.sqrt((x_sym-0.5)**2+(y_sym-0.5)**2)}).simplify()
u_nonsmooth_call = lambdify((x_sym, y_sym), u_nonsmooth)

# Symbolically calculate laplacians and gradients for both functions.
lap_u_harmonic = sympy.diff(sympy.diff(u_harmonic, x_sym), x_sym)+sympy.diff(sympy.diff(u_harmonic, y_sym), y_sym)
lap_u_harmonic_call = lambdify((x_sym, y_sym), lap_u_harmonic)

lap_u_nonsmooth = sympy.diff(sympy.diff(u_nonsmooth, x_sym), x_sym)+sympy.diff(sympy.diff(u_nonsmooth, y_sym), y_sym)
lap_u_nonsmooth_call = lambdify((x_sym, y_sym), lap_u_nonsmooth)

grad_u_x_harmonic = sympy.diff(u_harmonic, x_sym)
grad_u_y_harmonic = sympy.diff(u_harmonic, y_sym)
grad_u_x_harmonic_call = lambdify((x_sym, y_sym), grad_u_x_harmonic)
grad_u_y_harmonic_call = lambdify((x_sym, y_sym), grad_u_y_harmonic)

grad_u_x_nonsmooth = sympy.diff(u_nonsmooth, x_sym)
grad_u_y_nonsmooth = sympy.diff(u_nonsmooth, y_sym)
grad_u_x_nonsmooth_call = lambdify((x_sym, y_sym), grad_u_x_nonsmooth)
grad_u_y_nonsmooth_call = lambdify((x_sym, y_sym), grad_u_y_nonsmooth)

### Generate Training data using evenly spaced collocation points. f and u are functions evaluating the RHS and boundary values.
def generate_data(N, u, f):
  xp=jnp.linspace(0.,1.,N)
  yp=jnp.linspace(0.,1.,N)

  X, Y = jnp.meshgrid(xp, yp)

  coordinates = jnp.concatenate((jnp.reshape(X,(-1,1)),jnp.reshape(Y,(-1,1))),1)

  # the boundary snapshots locations are arranged linearly: bottom -> right -> top -> left
  # Xbdy and Ybdy are vectors of dimension 4*N

  xp_bdy=jnp.linspace(0.,1.,N)
  yp_bdy=jnp.linspace(1./N,1.-1./N,N-2)

  Xbdy = jnp.concatenate(
                  (jnp.concatenate((xp_bdy,jnp.full_like(yp_bdy,1))),
                   jnp.concatenate((xp_bdy,jnp.full_like(yp_bdy,0)))
                  )
                        )

  Ybdy = jnp.concatenate(
                  (jnp.concatenate((jnp.full_like(xp_bdy,0),yp_bdy)),
                   jnp.concatenate((jnp.full_like(xp_bdy,1),yp_bdy))
                  )
                        )

  coordinates_bdy = jnp.concatenate((jnp.reshape(Xbdy,(-1,1)),jnp.reshape(Ybdy,(-1,1))),1)

  # data values
  rhs_snapshots = jnp.reshape(-f(X, Y),(-1,1))
  bdy_snapshots = jnp.reshape(u(Xbdy,Ybdy),(-1,1))


  return coordinates, coordinates_bdy, rhs_snapshots, bdy_snapshots

### Define the Neural Network architecture.
# A helper function to randomly initialize weights and biases
def random_layer_params(m, n, key, scale=1e-2):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

# Initialize all layers for a deep neural network with width n and depth d
def init_deep_network_params(n, d, key):
  keys = random.split(key, d+1)
  sizes = [2] # 2 is space dimension
  for i in range(d):
    sizes.append(n)
  sizes.append(1)
  weights = []
  for m, n, k in zip(sizes[:-1], sizes[1:], keys):
    if m == n:
      weights.append(random_layer_params(m, n, k, jnp.sqrt(2) * jnp.power(2.0/15.0, 1.0/6.0) / jnp.sqrt(m * d))) # For ReLU3
    elif n == 1:
      weights.append(random_layer_params(m, n, k, 1.0 / jnp.sqrt(m)))
    else:
      weights.append(random_layer_params(m, n, k, 1.0))
  return weights

# Activation functions
def tanh(x):
  return (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))

def sigmoid(x):
  return 1.0 / (1.0 + jnp.exp(-x))

def relu3(x):
  return jnp.maximum(0,jnp.power(x,3))

# Calculate output based upon network parameters
def predict(params, values):
  activations = values
  init_w, init_b = params[0]
  outputs = jnp.dot(init_w,activations)
  outputs = outputs + init_b.reshape(outputs.shape)
  activations = tanh(outputs)
  for w, b in params[1:-1]:
    outputs = jnp.dot(w, activations)
    outputs = outputs + b.reshape(outputs.shape)
    activations = activations + relu3(outputs)

  final_w, final_b = params[-1]
  return (jnp.dot(final_w, activations) + final_b)

# Cast the output to a scalar.
def scalar_predict(params, values):
  return predict(params, values)[0]

# Calculate gradient of input
def grad_predict(params, values):
  return grad(scalar_predict, 1)(params, values)

# Calculate the Laplacian of the neural network function
def predict_laplacians(params, values):
  dxx = grad(lambda params, values: grad_predict(params, values)[0], 1)(params, values)[0]
  dyy = grad(lambda params, values: grad_predict(params, values)[1], 1)(params, values)[1]
  return dxx + dyy


# Enable evaluation at multiple points simultaneously
batched_predict = vmap(predict, in_axes=(None, 0))

# Enable evaluation of gradient at multiple points.
batched_grad_predict = vmap(grad_predict, in_axes=(None, 0))

# Enable evaluation of laplacian at multiple points simultaneously
batched_laplacians_predict = vmap(predict_laplacians, in_axes=(None, 0))

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
  loss_value, grads = value_and_grad(loss)(params, vals, vals_bdy, rhs_data, bdy_data, 1, 1)
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

def train_and_test(N, Ntest, u, lap_u, grad_u_x, grad_u_y, step_size, momentum, loss_type, step_count, decrease_interval, plot = False):
  # Initialize the network randomly.
  params = init_deep_network_params(width, depth, random.PRNGKey(0))

  # Generate the data based upon the harmonic example function.
  vals, vals_bdy, rhs_data, bdy_data = generate_data(N, u, lap_u)

  # Train the network based upon both the consistent and original loss.
  params = train(params, vals, vals_bdy, rhs_data, bdy_data, step_size, momentum, loss_type, step_count, decrease_interval)

  # Calculate and return the relative H1 error.
  xp_test=jnp.linspace(0.,1.,Ntest)
  yp_test=jnp.linspace(0.,1.,Ntest)

  X_test, Y_test = jnp.meshgrid(xp_test, yp_test)

  coordinates_test = jnp.concatenate((jnp.reshape(X_test,(-1,1)),jnp.reshape(Y_test,(-1,1))),1)

  nn_solution = batched_predict(params,coordinates_test)
  exact_solution = u(X_test,Y_test)

  if plot:
    plot_values(X_test,Y_test,jnp.reshape(nn_solution,jnp.shape(X_test)))
    plot_values(X_test,Y_test,exact_solution)

  # Calculate the H1 error.
  exact_grad_x = grad_u_x(X_test, Y_test)
  exact_grad_y = grad_u_y(X_test, Y_test)
  exact_grads = jnp.concatenate((jnp.reshape(exact_grad_x, (-1,1)), jnp.reshape(exact_grad_y, (-1,1))), axis = 1)
  nn_grads = batched_grad_predict(params, coordinates_test)

  solution_norm = (1.0/Ntest)*jnp.linalg.norm(exact_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(exact_solution)
  error = (1.0/Ntest)*jnp.linalg.norm(exact_grads - nn_grads, 'fro') + (1.0/Ntest)*jnp.linalg.norm(jnp.reshape(nn_solution, jnp.shape(X_test)) - exact_solution)

  return error / solution_norm
### Run the experiments.
for N in Nlist:
  print('Number of collocation points in each direction: %d' % N)
  
  error = train_and_test(N, Ntest, u_harmonic_call, lap_u_harmonic_call, grad_u_x_harmonic_call, grad_u_y_harmonic_call, step_size, momentum, 'original', step_count, decrease_interval) 
  print('Using the original loss function for the harmonic u gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, u_harmonic_call, lap_u_harmonic_call, grad_u_x_harmonic_call, grad_u_y_harmonic_call, step_size, momentum, 'consistent', step_count, decrease_interval) 
  print('Using the consistent loss function for the harmonic u gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, u_nonsmooth_call, lap_u_nonsmooth_call, grad_u_x_nonsmooth_call, grad_u_y_nonsmooth_call, step_size, momentum, 'original', step_count, decrease_interval) 
  print('Using the original loss function for the nonsmooth u gives a relative error of: %lf' % error)
  
  error = train_and_test(N, Ntest, u_nonsmooth_call, lap_u_nonsmooth_call, grad_u_x_nonsmooth_call, grad_u_y_nonsmooth_call, step_size, momentum, 'consistent', step_count, decrease_interval) 
  print('Using the consistent loss function for the nonsmooth u gives a relative error of: %lf' % error)
