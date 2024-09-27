# Author: Jonathan Siegel and Andrea Bonito
#
# Generates the point sample data and true solution for both elliptic examples of consistent PINNs.

import jax.numpy as jnp
import sympy
from sympy import lambdify

def generate_coordinate_grid(N, a, b, dim):
  """Generates a coordinate grid with N points in each direction on [a,b]^(dim).

  args:
    N: Number of grid points in each direction
    a: lower bound
    b: upper bound
    dim: dimension of the space

  returns:
    coordinates: the coordinates of the sample points
    coordinates_bdy: the coordinates of the boundary sample points
  """
  coordinate_points = []
  for i in range(dim):
    coordinate_points.append(jnp.linspace(a,b,N))
  coordinate_tuple = jnp.meshgrid(*coordinate_points)
  return jnp.concatenate(list(map(lambda x: jnp.reshape(x,(-1,1)), coordinate_tuple)), 1)


def generate_coordinates_cube(N, a, b, dim):
  """Generates coordinates for interior and boundary points with N points in each direction on [a,b]^(dim).

  args:
    N: Number of grid points in each direction
    a: lower bound
    b: upper bound
    dim: dimension of the space

  returns:
    coordinates: the coordinates of the sample points
    coordinates_bdy: the coordinates of the boundary sample points
  """
  coordinates = generate_coordinate_grid(N, a, b, dim)
  bdy_indices , = jnp.any(((coordinates == a) | (coordinates == b)), axis = 1).nonzero()
  coordinates_bdy = jnp.take(coordinates, bdy_indices, 0)
  return coordinates, coordinates_bdy

def generate_2d_poisson_experiment(N, Ntest, exp_type):
  """Generates the point sample data and true solution data for the 2d Poisson test problems.

  args:
    N: Number of collocation points in each direction for training
    Ntest: Number of points in each direction for true solution
    exp_type: type of experiment to generate, either `harmonic', `nonsmooth', or `smooth'

  returns:
    coordinates: x and y coordinates of the RHS data points
    coordinates_bdy: x and y coordinates of the boundary data
    coordinates_test: x and y coordinates of the testing collocation points
    rhs_data: RHS data values
    bdy_data: boudnary data values
    solution: values of the true solution at the test coordinates
    solution_grad_x: x gradients of the true solution at the test coordinates
    solution_grad_y: y gradients of the true solution at the test coordinates.
  """
  x_sym = sympy.Symbol('x')
  y_sym = sympy.Symbol('y')
  r_sym = sympy.Symbol('r', nonnegative=True)

  # Select either the harmonic or nonsmooth experiment.
  if exp_type == 'harmonic':  
    u = sympy.exp(x_sym)*sympy.cos(sympy.pi*y_sym)
    u_call = lambdify((x_sym, y_sym), u)
  elif exp_type == 'smooth':
    u = sympy.exp(2.0*(x_sym + y_sym))*sympy.cos(2.0*sympy.pi*(y_sym - x_sym))/(1.0 + 8.0*x_sym**2 + y_sym**2)
    u_call = lambdify((x_sym, y_sym), u) 
  else:
    u_tmp = 1000*x_sym*(1-x_sym)*y_sym*(1-y_sym)*r_sym**(4.5)
    u = u_tmp.subs({r_sym:sympy.sqrt((x_sym-0.5)**2+(y_sym-0.5)**2)}).simplify()
    u_call = lambdify((x_sym, y_sym), u)

  # Construct negative laplacian of solution.
  lap_u = -1.0*(sympy.diff(sympy.diff(u, x_sym), x_sym)+sympy.diff(sympy.diff(u, y_sym), y_sym))
  lap_u_call = lambdify((x_sym, y_sym), lap_u)

  # Construct gradient of solution.
  grad_u_x = sympy.diff(u, x_sym)
  grad_u_y = sympy.diff(u, y_sym)
  grad_u_x_call = lambdify((x_sym, y_sym), grad_u_x)
  grad_u_y_call = lambdify((x_sym, y_sym), grad_u_y)

  # Generate training data.
  coordinates, coordinates_bdy = generate_coordinates_cube(N, 0., 1., 2)

  # data values
  rhs_data = lap_u_call(coordinates[:,0], coordinates[:,1])
  bdy_data = u_call(coordinates_bdy[:,0], coordinates_bdy[:,1])

  # Generate solution data.
  coordinates_test = generate_coordinate_grid(Ntest, 0., 1., 2)
  X_test = coordinates_test[:,0]
  Y_test = coordinates_test[:,1]
  solution = u_call(X_test, Y_test)
  solution_grad_x = grad_u_x_call(X_test,Y_test)
  solution_grad_y = grad_u_y_call(X_test,Y_test)
  solution_grads = jnp.column_stack((solution_grad_x, solution_grad_y))

  return coordinates, coordinates_bdy, coordinates_test, rhs_data, bdy_data, solution, solution_grads

def generate_3d_poisson_experiment(N, Ntest, exp_type):
  """Generates the point sample data and true solution data for the 3d Poisson test problems.

  args:
    N: Number of collocation points in each direction for training
    Ntest: Number of points in each direction for true solution
    exp_type: type of experiment to generate, currently only option is `smooth'

  returns:
    coordinates: x and y coordinates of the RHS data points
    coordinates_bdy: x and y coordinates of the boundary data
    coordinates_test: x and y coordinates of the testing collocation points
    rhs_data: RHS data values
    bdy_data: boudnary data values
    solution: values of the true solution at the test coordinates
    solution_grad_x: x gradients of the true solution at the test coordinates
    solution_grad_y: y gradients of the true solution at the test coordinates.
  """
  x_sym = sympy.Symbol('x')
  y_sym = sympy.Symbol('y')
  z_sym = sympy.Symbol('z')

  # Currently the only experiment is smooth.
  u = sympy.exp(x_sym + y_sym + z_sym)*sympy.cos(3.0*sympy.pi*(y_sym - x_sym + z_sym))/(1.0 + x_sym**2 + y_sym**2 + z_sym**4)
  u_call = lambdify((x_sym, y_sym, z_sym), u)

  # Construct negative laplacian of solution.
  lap_u = -1.0*(sympy.diff(sympy.diff(u, x_sym), x_sym)+sympy.diff(sympy.diff(u, y_sym), y_sym)+sympy.diff(sympy.diff(u, z_sym), z_sym))
  lap_u_call = lambdify((x_sym, y_sym, z_sym), lap_u)

  # Construct gradient of solution.
  grad_u_x = sympy.diff(u, x_sym)
  grad_u_y = sympy.diff(u, y_sym)
  grad_u_z = sympy.diff(u, z_sym)
  grad_u_x_call = lambdify((x_sym, y_sym, z_sym), grad_u_x)
  grad_u_y_call = lambdify((x_sym, y_sym, z_sym), grad_u_y)
  grad_u_z_call = lambdify((x_sym, y_sym, z_sym), grad_u_z)
  
  # Generate training data.
  coordinates, coordinates_bdy = generate_coordinates_cube(N, 0., 1., 3)

  # data values
  rhs_data = lap_u_call(coordinates[:,0], coordinates[:,1], coordinates[:,2])
  bdy_data = u_call(coordinates_bdy[:,0], coordinates_bdy[:,1], coordinates_bdy[:,2])

  # Generate solution data.
  coordinates_test = generate_coordinate_grid(Ntest, 0., 1., 3)
  X_test = coordinates_test[:,0]
  Y_test = coordinates_test[:,1]
  Z_test = coordinates_test[:,2]
  solution = u_call(X_test, Y_test, Z_test)
  solution_grad_x = grad_u_x_call(X_test,Y_test, Z_test)
  solution_grad_y = grad_u_y_call(X_test,Y_test, Z_test)
  solution_grad_z = grad_u_z_call(X_test,Y_test, Z_test)
  solution_grads = jnp.column_stack((solution_grad_x, solution_grad_y, solution_grad_z))

  return coordinates, coordinates_bdy, coordinates_test, rhs_data, bdy_data, solution, solution_grads

