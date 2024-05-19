# Consistent Physics Informed Neural Networks

This repository contains a demonstration of the consistent PINNs methodology for solving PDEs using neural networks, which designs a loss function by discretizing the Sobolev norms appearing in the PDE regularity theory. A detailed description of the method can be found in the paper:

Forthcoming, to be added soon

## Elliptic PDEs

Consider the Poisson equation:

Delta u = f on Omega
u = g on the boundary of Omega

W solve this equation using both the consistent PINNs and the original PINNs loss function. We test two examples, one where f = 0, i.e. u is harmonic, with exact solution given by

u(x) = e^x * cos(pi * y),

and the second, where g = 0, with exact solution given by

u(x) = 1000 * x * (1-x) * y * (1-y) * (x^2 + y^2)^(9/4).

This second example was specifically chosen so that the RHS f is not smooth. Based upon our experiments, using the consistent PINNs loss function results in errors which are about 3-5 times smaller than when using the original PINNs loss function, despite comparable final loss values attained for both methods. Running the Python script **elliptic-pde-experiments.py** reproduces our experimental results.
