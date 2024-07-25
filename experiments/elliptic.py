# Author: Jonathan Siegel and Andrea Bonito
#
# Generates the point sample data and true solution for both elliptic examples of consistent PINNs.

import jax.numpy as jnp
import sympy
from sympy import lambdify

def generate_data(N, u, f):
  """Generates data given symbolic functions u and f for the boundary values and RHS respectively.

  args:
    N: Number of grid points in each direction
    u: function to be evaluated on the boundary
    f: RHS of the Poisson equation

  returns:
    coordinates: the x and y coordinates of the sample points
    coordinates_bdy: the x and y coordinates of the boundary sample points
    rhs_values: the value of the RHS at the coordinates
    bdy_values: the values of the boundary data at coordinates_bdy
  """
  xp=jnp.linspace(0.,1.,N)
  yp=jnp.linspace(0.,1.,N)

  X, Y = jnp.meshgrid(xp, yp)

  coordinates = jnp.concatenate((jnp.reshape(X,(-1,1)),jnp.reshape(Y,(-1,1))),1)

  # the boundary locations are arranged linearly: bottom -> right -> top -> left
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
  rhs_data = jnp.reshape(f(X, Y),(-1,1))
  bdy_data = jnp.reshape(u(Xbdy,Ybdy),(-1,1))

  return coordinates, coordinates_bdy, rhs_data, bdy_data

def generate_elliptic_experiment(N, Ntest, exp_type):
  """Generates the point sample data and true solution for the harmonic test problem.

  args:
    N: Number of collocation points in each direction for training
    Ntest: Number of points in each direction for true solution
    exp_type: type of experiment to generate, either `harmonic' or `nonsmooth'

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
  coordinates, coordinates_bdy, rhs_data, bdy_data = generate_data(N, u_call, lap_u_call)

  # Generate solution data.
  xp_test=jnp.linspace(0.,1.,Ntest)
  yp_test=jnp.linspace(0.,1.,Ntest)

  X_test, Y_test = jnp.meshgrid(xp_test, yp_test)

  coordinates_test = jnp.concatenate((jnp.reshape(X_test,(-1,1)),jnp.reshape(Y_test,(-1,1))),1)

  solution = u_call(X_test,Y_test)
  solution_grad_x = grad_u_x_call(X_test,Y_test)
  solution_grad_y = grad_u_y_call(X_test,Y_test)
  solution_grads = jnp.concatenate((jnp.reshape(solution_grad_x, (-1,1)), jnp.reshape(solution_grad_y, (-1,1))), axis = 1)

  return coordinates, coordinates_bdy, coordinates_test, rhs_data, bdy_data, solution, solution_grads

