# Consistent Physics Informed Neural Networks

This repository contains a demonstration of the consistent PINNs methodology for solving PDEs using neural networks, which designs a loss function by discretizing the Sobolev norms appearing in the PDE regularity theory. A detailed description of this method can be found in the paper:

- Andrea Bonito, Ronald DeVore, Guergana Petrova and Jonathan W. Siegel. "[Convergence and error control of consistent PINNs for elliptic PDEs.](https://arxiv.org/abs/2406.09217)" arXiv preprint arXiv:2406.09217 (2024).

## Elliptic PDEs

Consider the Poisson equation:

$-\Delta u = f$ on $\Omega$

$u = g$ on $\partial \Omega$

We solve this equation using a variety of loss functions in the PINNs formulation on three test problems in 2d. These experiments can be reproduced by running the Python script **poisson-2d-nat-grad.py**. The results of the experiments and a corresponding discussion can be found in the aforementioned paper.

# Citation

    @article{bonito2024convergence,
      title={Convergence and error control of consistent PINNs for elliptic PDEs}, 
      author={Andrea Bonito and Ronald DeVore and Guergana Petrova and Jonathan W. Siegel},
      journal={arXiv preprint arXiv:2406.09217},
      year={2024},
    }

## Natural Gradient Newton Optimizer

We use the natural newton optimizer to train the PINNs for each loss function in our experiments. This optimizer allows us to obtain the same solution accuracy much more efficiently using a much smaller network than gradient descent. The optimizer we have implemented is based upon the paper:

    @inproceedings{muller2023achieving,
      title={Achieving high accuracy with PINNs via energy natural gradient descent},
      author={M{\"u}ller, Johannes and Zeinhofer, Marius},
      booktitle={International Conference on Machine Learning},
      pages={25471--25485},
      year={2023},
      organization={PMLR}
    }
