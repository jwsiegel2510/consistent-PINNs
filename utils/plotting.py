# Author: Jonathan Siegel and Andrea Bonito
#
# Contains utility functions for plotting PINNs solutions.

import matplotlib.pyplot as plt
from matplotlib import cm
import jax.numpy as jnp

# Function for plotting the solution.
def plot_values(Xval, Yval, solution):
  """A function which plots the solution evaluated on a grid.
  
  Args:
    Xval: An array of x-coordinates
    Yval: An array of y-cordinates
    solution: An array of solution values
  """
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  surf = ax.plot_surface(Xval, Yval, solution, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  plt.show()
