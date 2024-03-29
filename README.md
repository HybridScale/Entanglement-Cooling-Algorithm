[![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)](./LICENSE) [![DOI](https://zenodo.org/badge/349075288.svg)](https://zenodo.org/record/7252232#.Y7yNv6fMJhG) [![DOI](https://img.shields.io/badge/DOI-10.1145%2F3313828%20-orange)](https://doi.org/10.48550/arXiv.2210.13495) [![DOI](https://img.shields.io/badge/DOI-10.1002%2Fcpe.3394%20-orange)](http://dx.doi.org/10.23919/mipro55190.2022.9803591)

# Entanglement cooling algorithm

The **Entanglement Cooling** is a simulated annealing Metropolis Monte-Carlo algorithm where local unitaries are stochastically applied to a quantum spin state. The unitaries are sampled uniformly from two different sets (the non-universal and universal) and applied to any two neighboring spins. The cost function of this procedure is an entanglement measure which favors the acceptance and therefore the application of local unitaries which remove/destroy entanglement between neighboring spins. The initial states are the ground states of the one-dimensional transverse-field Ising model in its different macroscopic phases (paramagnetic, magnetically ordered, and topologically frustrated).

The code is written in Python, which allows for straightforward portability across different computing platforms. The code supports execution on shared memory systems (manycore and multi-CPU systems) and on distributed GPU systems using the Message Passing Interface (MPI). The code currently supports only the NVIDIA GPUs.

The code was developed at the [Ruđer Bošković Institute](https://www.irb.hr/) by the Condensed Matter and Statistical Physics Group ([Q-Team](http://thphys.irb.hr/qteam/)) and the [Centre for Informatics and Computing](https://cir.com.hr/) and supported by the Croatia Science Foundation through project UIP-2020-02-4559 "Scalable High-Performance Algorithms for Future Heterogeneous Distributed Computing Systems"([HybridScale](https://www.croris.hr/projekti/projekt/6243?lang=en)).


## Dependencies

- numpy
- scipy
- mpi4py
- cupy

A detailed list of the required Python libraries and dependencies is given in the file [Requirements.txt](./Requirements.txt).

## Versions of the code

The Entaglement cooling algorithm supports three execution modes depending on the available hardware resources:

- **CPU:** Computes a Monte Carlo simulation using all available cores and processors of the shared memory system. If the code is started with MPI (mpirun), the number of Monte Carlo simulations started is equal to the number of MPI ranks.

- **GPU:** Calculates a Monte Carlo simulation on a GPU. If the code is started with MPI (mpirun), the number of Monte Carlo simulations started is equal to the number of MPI ranks. Several MPI ranks can be connected to the same GPU.

- **batchedGEMM:** Calculates a number of Monte Carlo simulations on a GPU. The Monte Carlo simulations are batched and run on a GPU. When the code is started with MPI (mpirun), the number of Monte Carlo simulations is equal to the product of the MPI ranks and the batch size.

## Quick start

### Cloning the code

```bash
git clone https://github.com/HybridScale/Entanglement-Cooling-Algorithm.git
```

### Install Python environment

The repository provides the file [Requirements.txt](./Requirements.txt) with Python packages needed for simulation. To create new Virtual Envirnoment in Python with Conda and install all required packages run:

```bash
conda create --name myenv --file Requirements.txt
```

To activate conda environment with all the required dependencies installed run:
```bash
conda activate myenv
```

### Running

The repository provides the script [src/main.py](main.py), which implements the CLI interface.  
To get information about different code versions, run:

```bash
python src/main.py -h
``` 
There are three different versions available: `CPU`, `GPU` and `BatchedGEMM`.
 
Both `GPU` and `BatchedGEMM` support execution on GPUs. With `GPU`, one Monte Carlo simulation is executed per GPU and with `batchedGEMM`, multiple simulations are combined in a batch operation and executed simulatenously on a single GPU.

The simulation can be resumed from the last simulation with the same initial parameters or a new simulation can be started.
To check the options, execute:

```bash
python src/main.phy {CPU,GPU,batchedGEMM} -h
```

Use positional arguments to select the desired version. CLI also provides information about all available command line options and parameters.

```bash
python src/main.phy {CPU, GPU, batchedGEMM} {new, resume} -h
```

## Examples
In all the following examples we use the same simulation parameters: **19** lattice sites, subsystem of size **9**, coupling parameter **2.5**, **1000** Monte Carlo simulation steps with **10000000** steps finally calculated.

### CPU version
Starting a simulation takes up all the available computing cores and processors of a computer.

```bash 
python src/main CPU new --N 19 --R 9 --L 2.5 -- MC 1000 -w 10000000
```

Multiple Monte Carlo simulations can be computed simultaneously with `MPI` by starting `N` simulations, but all `MPI` processes would compete for resources (CPU cores). To work around this problem, set the number of cores that can be used per simulation (i.e. per MPI rank).
If your system has 96 cores and you want to run 8 Monte Carlo simulations, you must set the number of threads per MPI rank (`OMP_NUM_ THREADS`) to 12.

```bash 
export OMP_NUM_THREADS = 12 
mpirun -n 8 python src/main CPU new --N 19 --R 9 --L 2.5 -- MC 1000 -w 10000000
```

For best performance, the total number of cores must be evenly distributed among the MPI ranks.
If you have 4 compute nodes with 64 cores each and want to run 16 Monte Carlo simulations, you should divide 256 (total number of nodes) by 16 (number of MPI ranks, i.e. Monte Carlo simulations).

```bash 
export OMP_NUM_THREADS = 16 
mpirun -n 16 python src/main CPU new --N 19 --R 9 --L 2.5 -- MC 1000 -w 10000000
```

The above code distributes the Monte Carlo simulations evenly (4 simulations per node) and allocates 16 computer cores for each simulation.

### GPU version
Run a Monte Carlo simulation on a single GPU:

```bash 
mpirun -n 4 python src/main.py GPU new --N 19 --R 9 --L 2.5 -- MC 1000 -w 10000000
```

Multiple simulations can be computed simultaneously using `MPI` to run `N` simulations that distribute the `MPI` process across multiple GPUs present in the system. If you start 12 simulations on a system with 4 GPUs, 3 simulations will be run per GPU and each simulation will create its own process context, resulting in competition for resources. To solve this problem, start NVIDIA Multi-Process Service ([MPS](https://docs.nvidia.com/deploy/mps/index.html)) before running the simulations and stop it after running.

```bash 
nvidia-cuda-mps-control -d
sleep 10
mpirun -n 4 python src/main.py GPU new --N 19 --R 9 --L 2.5 -- MC 1000 -w 10000000 echo quit | nvidia-cuda-mps-control
```

### batchedGEMM version
The number of simulations computed on a single GPU is specified with the parsing argument `--bs`. To set a run with 8 simulations on a GPU, execute:

```bash 
python src/main.py batchedGEMM new --N 19 --R 9 --L 2.5 -- MC 1000 -w 10000000 --bs 8
```

There is an upper limit to the number of simulations that can be run on a single graphics card, so it is possible to run multiple `batchedGEMMs` simultaneously. It is recommended to run a single batchedGEMM version per GPU. In the next example, 4 `batchedGEMMs` are run on a system with 4 GPUs, with each `batchedGEMM` running 8 simulations for a total of 32 Monte Carlo simulations:

```bash 
mpirun -n 4 python src/main.py batchedGEMM new --N 19 --R 9 --L 2.5 -- MC 100 -w 10000000 --bs 8
```
### Resume simulation
After the simulation is started with `new`, the configuration file and simulation states are saved in the folder specified with the optional input parameter `--f` (default folder name `saved_sates_N_R_L`). Using the `resume` argument and specifying the folder where the configuration file and simulation states are located, the entire simulation is resumed from the last step. With the parameter `--MC` can be selected how many additional steps to the desired steps selected from `new` should be calculated. Continuation of simulation should be done according to the number of Monte Carlo simulations already calculated. If you want to continue the simulation with 32 Monte Carlo simulations, you can do this on the `CPU`/`GPU` versions:

```bash
mpirun -n 32 python src/main.py {CPU,GPU} resume saved_states_19_9_2.5 --MC 10000
```
or in the `batchedGEMM` version with 8 Monte Carlo simulation per GPU:
```
mpirun -n 4 python src/main bathcedGEMM resume saved_states_19_9_2.5 --MC 10000 --bs 8

```

## Fine-tune the execution

To be done

## Developers

### The main code

- [Jovan Odavić](https://github.com/dzovan137) 
- Gianpaolo Torre
- Fabio Franchini
- Salvatore Marco Gaimpaolo

### GPU parallelisation and code optimisation

- [Davor Davidović](https://github.com/ddavidovic)
- [Nenad Mijić](https://github.com/Nenad03)

## Contribution

This repository mirrors the principal repository of the code on Bitbucket. If you want to contribute to the code please contact *davor.davidovic@irb.hr* or *jovan.odavic@irb.hr*.

## How to cite the code

Description of the computational model and the entanglement cooling with Metropolis Monte Carlo:

- J. Odavić, G. Torre, N. Mijić, D. Davidović, F. Franchini, S. M. Giampaolo.  *Random unitaries, Robustness, and Complexity of Entanglement*, Quantum 7, 1115 (2023), [doi: 10.22331/q-2023-09-15-1115](https://doi.org/10.22331/q-2023-09-15-1115) ([pdf](https://quantum-journal.org/papers/q-2023-09-15-1115/))

The code and parallelisation on distributed multi-GPU computing architectures:

- N. Mijic, D. Davidovic. *Batched matrix operations on distributed GPUs with application in theoretical physics.* 45th Jubilee International Convention on Information, Communication and Electronic Technology (MIPRO), 2022, pp. 293-299, [doi: 10.23919/MIPRO55190.2022.9803591](http://dx.doi.org/10.23919/mipro55190.2022.9803591). ([pdf](http://fulir.irb.hr/7514/)).

## Copyright and License

This code is published under GNU General Public License v3.0 ([GNU GPLv3](./LICENSE))
