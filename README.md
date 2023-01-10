[![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)](./LICENSE) [![DOI](https://zenodo.org/badge/349075288.svg)](https://zenodo.org/record/7252232#.Y7yNv6fMJhG) [![DOI](https://img.shields.io/badge/DOI-10.1145%2F3313828%20-orange)](https://doi.org/10.48550/arXiv.2210.13495) [![DOI](https://img.shields.io/badge/DOI-10.1002%2Fcpe.3394%20-orange)](http://dx.doi.org/10.23919/mipro55190.2022.9803591)

# Entanglement cooling algorithm

The **Entanglement cooling algorithm** computes the statistical properties of the entanglement spectrum by applying a Metropolis-like entanglement cooling algorithm generated by different sets of local gates, on states sharing the same statistic. The initial states are the ground states of the one-dimensional quantum Ising model in its different macroscopic phases (paramagnetic, the magnetically ordered and the topologically frustrated).

The code is developed at the [Ruđer Bošković Institute](https://www.irb.hr/) by the Condensed Matter and Statistical Physics Group and the [Centre for Informatics and Computing](https://cir.com.hr/)

The code is written in Python which enables straightforward portability over different computational platforms. The code supports execution on shared memory systems (manycore and multi-CPU systems) and on distributed GPUs systems using the Message Passing Interface (MPI).

## Prerequisits

The list of required Python libraries and other prerequisits is given in the file [Requirements.txt](./Requirements.txt)

- numpy
- scipy
- mpi4py (if distributed GPU support is required)
- cupy (if GPU support is required)
- ???

## Quick start

### Cloning the code

To be added once 

### Building and installing

### Execution environment

### Running

### Examples

## Fine-tune the execution

Describe how to choose an optimaln number of MPI ranks per GPU and the number of GPU w.r.t to the problem size (number of spins) and the number of MC trajectories.

## Developers

### Developing the main code

- Jovan Odavić 
- Gianpaolo Torre
- Fabio Franchini
- Salvatore Marco Gaimpaolo

### GPU parallelisation and code optimisation

- Davor Davidović
- Nenad Mijić

## Contribution

This repository mirrors the principal repository of the code on Bitbucket. If you want to contribute to the code please contact davor.davidovic@irb.hr or jovan.odavic@irb.hr

## How to cite the code

Description of the computational model and the entanglement cooling with Metropolis Monte Carlo:

J. Odavić, G. Torre, N. Mijić, D. Davidović, F. Franchini, S. M. Giampaolo.  *Random unitaries, Robustness, and Complexity of Entanglement*. [arXiv:2210.13495](https://doi.org/10.48550/arXiv.2210.13495)

The code and parallelisation on distributed multi-GPU computing architectures:

N. Mijic, D. Davidovic. *Batched matrix operations on distributed GPUs with application in theoretical physics.* 45th Jubilee International Convention on Information, Communication and Electronic Technology (MIPRO), 2022, pp. 293-299, [doi: 10.23919/MIPRO55190.2022.9803591](http://dx.doi.org/10.23919/mipro55190.2022.9803591).


The full paper is available [here](http://fulir.irb.hr/7514/).

## Copyright and License

This code is published under GNU General Public License v3.0 ([GNU GPLv3](./LICENSE))