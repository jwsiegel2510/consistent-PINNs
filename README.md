# Consistent Physics Informed Neural Networks

This repository contains a demonstration of the consistent PINNs methodology for solving PDEs using neural networks, which designs a loss function by discretizing the Sobolev norms appearing in the PDE regularity theory. A detailed description of this method can be found in the paper:

- Andrea Bonito, Ronald DeVore, Guergana Petrova and Jonathan W. Siegel. "[Convergence and error control of consistent PINNs for elliptic PDEs.](https://arxiv.org/abs/2406.09217)" arXiv preprint arXiv:2406.09217 (2024).

## Elliptic PDEs

Consider the Poisson equation:

$-\Delta u = f$ on $\Omega$

$u = g$ on $\partial \Omega$

We solve this equation using both the consistent PINNs and the original PINNs loss function. We test two examples, one where $f = 0$, i.e. u is harmonic, with exact solution given by

$u(x) = e^x\cos(\pi y),$

and the second, where $g = 0$, with exact solution given by

$u(x) = 1000x(1-x)y(1-y)((x - 1/2)^2 + (y - 1/2)^2)^{9/4}.$

This second example was specifically chosen so that the RHS f is not smooth and scaled so that its $L_\infty$-norm is about $1$. 

Based upon our experiments, using the consistent PINNs loss function results in errors which are about 3-5 times smaller than when using the original least squares PINNs loss function. Running the Python script **elliptic-pde-experiments.py** reproduces our experimental results.

# Citation

    @article{bonito2024convergence,
      title={Convergence and error control of consistent PINNs for elliptic PDEs}, 
      author={Andrea Bonito and Ronald DeVore and Guergana Petrova and Jonathan W. Siegel},
      journal={arXiv preprint arXiv:2406.09217},
      year={2024},
    }
