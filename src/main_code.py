"""
Model:  Transverse Ising spin chain
Method: Simulated annealing (Monte Carlo procedure with decreasing fictitious temperature) 
using tensor networks base representation performing the Exact Diagonalization and all remaining operations.
Goal: Computation of the ground state von Neuman entropy (Entanglement entropy) and attempting to disentagle 
all entaglement using local gates.

Study: 
- Increase Nsites while keeping odd (but also keep in mind R to be almost half of the system) see where we end up
- Change lambdaa and see where we end up
- Increase MC statistics
- Increase MC simulation duration (are we converging?)
- Keeping acceptence rate?

"""

import numpy as np
from itertools import combinations
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, eigsh
import math
from numpy import linalg as LA
from math import e

import sys
#sys.path.append("../../dependencies")
from dependencies import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('apsRMT.mplstyle')


# this dumb function doesn't work when in dependecies.py file so we put it here
def doApplyHamClosed(psiIn):
    """ supplementary function  cast the Hamiltonian 'H' as a linear operator """
    return doApplyHamTENSOR(psiIn, hloc, Nsites, usePBC)




# model parameters
Nsites    = 7               # number of lattice sites
usePBC    = True            # use periodic or open boundaries (we typically want PBC)
R         = 4               # subsystem size
lambdaa   = 2.5             # coupling parameter (for odd number of Nsites )


# simulation parameters
numval    = 1               # number of eigenstates to compute (we want ground state so =1 is what we want)
dt        = np.pi/10.0      # 'time-evolution' time step (this is a parameter we should change a bit to see if the results are consistent with it)
MCsteps   = 1000            # Monte Carlo steps
M         = 10              # number of repetitions of the Monte Carlo procedure


# defining a logaritmically decreasing temperature grid 
T_grid    = np.logspace(-4,-8, num=101,base=10)
print('T_grid =', T_grid)

sX, sY, sZ, sI = PauliMatrice()

concurrence1 = []
concurrence2 = []
magz_exact = []
magz_num = []


# IMMEDIATELY PLOTTING ! 
fig = plt.figure()
grid = plt.GridSpec(1, 1, wspace=0.4, hspace=0.3)

ax = fig.add_subplot(grid[0, :1])

# defining the local Hamiltonian and solving the eigenvalue problem to obtain the ground state wavefunction
hloc = tfim_LocalHamiltonian_new(lambdaa)
H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)

eigenvalues, eigenvectors = eigsh(H, k=numval, which='SA',return_eigenvectors=True)

idx = eigenvalues.argsort()[::1]   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]

# computing the initial von Neumann entropy
ent, ee_list = vonNeumann(Nsites,R,eigenvectors[:,0])

concurrence1.append(ent)

# generate the list of pairs for the application of the local Hamiltonian
list_pairs_local_ham = generate_dictionary(Nsites,1, True)

lista_x   = np.zeros(MCsteps)
lista_y   = np.zeros((MCsteps,M))
list_acc  = np.zeros(M)

# loop over different monte carlo realizations
for yyy in range(0,M):

    state = eigenvectors[:,0].copy()
    ent_old = ent.copy()
    ee_list_old = ee_list.copy()

    accepted = 0

    cc = 0
    # monte carlo sampling
    for yy in range(0,MCsteps):

        # select 2 contiguous spins to which the the selected gate is applied
        random_pair, random_pair_id = select_cooling_evolution_indices(list_pairs_local_ham)
        random_gate, random_gate_id = select_cooling_evolution_gates()

        # apply the local gate
        new_state = ApplyLocalGate(random_gate,random_pair,state,Nsites,2,dt) 

        # compute the new von Neuman entropy
        ent_new, ee_list_new, relevant_partitions = vonNeumann_aftergate_correct(Nsites,R,new_state,random_pair)
        ent_tamp = ee_list_old.copy()

        # rewrite the array with the two different entropies
        ent_tamp[relevant_partitions[0]] = ee_list_new[0] 
        ent_tamp[relevant_partitions[1]] = ee_list_new[1] 

        old_value = -np.average(ee_list_old)
        new_value = -np.average(ent_tamp)

        # we avoid overflow
        p_exp = -(1.0/T_grid[cc])*(new_value - old_value)
        if (p_exp > 50):
            p = 1.0
        else:    
            p = np.exp(p_exp)

        # make the metropolis step of choice
        random_value = np.random.uniform(0,1)
        if (random_value <= min(1.0,p)):
            accepted += 1
            state = new_state.copy()
            ee_list_old = ent_tamp.copy()

     
        lista_x[yy] = yy
        lista_y[yy,yyy] = -np.average(ee_list_old)

        print(lambdaa,yy,accepted, T_grid[cc],old_value,new_value, p, random_gate_id, random_pair,random_value)
            
        if (yy % int(MCsteps/100) == 0):
            cc += 1


    list_acc[yyy] = accepted
    
concurrence2.append(ent_new)

lista_y_aver = []
for yy in range(0,MCsteps):
    aa = 0.0
    for yyy in range(0,M): 
        aa = aa + lista_y[yy,yyy]
        
    lista_y_aver.append( aa/M )



ax.plot(lista_x,lista_y_aver,label=r'$\lambda = %f $, nb acc = %i' % (lambdaa,np.average(accepted)))







ax.set_xlabel(r' MC step',fontsize=30)
ax.set_ylabel(r' von Neumann entropy ',fontsize=30)
ax.grid(True,alpha=0.2)
ax.tick_params( which='major',bottom=True,top=True,left=True,right=True,labeltop=False,labelbottom=True,labelsize='18')
ax.legend(ncol=1,loc='best',fontsize='15')
ax.set_xscale('log')

plt.savefig("von_Neumann_cooling_temperature.png", dpi=300)
plt.show()    

