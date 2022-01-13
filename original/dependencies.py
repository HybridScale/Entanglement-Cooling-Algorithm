# needed libraries
import numpy as np
from scipy.integrate import quad
import cmath
from collections import OrderedDict
import itertools
from scipy import linalg
from numpy import linalg as LA
from scipy.linalg import sqrtm
import math

############################# DEFINING THE HAMILTONIANS ##########################


def tfim_LocalHamiltonian(J,h):
    """     
    From: ' Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 
    """

    sX, sY, sZ, sI = PauliMatrice()

    hloc = (-(J/4.0)*np.kron(sX, sX) + (h/2.0)*np.kron(sZ,sI) ).reshape(2, 2, 2, 2)

    return hloc


def tfim_LocalHamiltonian_new(lambdaa):

    """ Trasverse Ising Hamiltonian - Fabio/Marco """ 

    sX, sY, sZ, sI = PauliMatrice()

    #hloc = (-lambdaa*np.kron(sX, sX) - 0.5*(np.kron(sZ,sI) + np.kron(sI,sZ)) ).reshape(2, 2, 2, 2)
    hloc = (lambdaa*np.kron(sX, sX) - np.kron(sZ,sI) ) .reshape(2, 2, 2, 2)
    return hloc    



def XYchainDM_LocalHamiltonian(J,gamma,D):

    """ Dzyalozinski- Morya Hamiltonian - Petar Mali paper"""

    sX, sY, sZ, sI = PauliMatrice()

    hloc = ((J/4.0)*(1.0 + gamma)*np.kron(sX, sX) - (J/4.0)*(1.0 - gamma)*np.kron(sY, sY) +  ((D*J)/4.0)*(np.kron(sX, sY) + np.kron(sY, sX))).reshape(2, 2, 2, 2)

    return hloc

def FabioMarcoFrustrated_LocalHamiltonian(delta,phi):

    """ Fabio/Marco Hamiltonian"""


    sX, sY, sZ, sI = PauliMatrice()

    local_parity_operator = (np.kron(sI,sI) + np.kron(sZ,sZ) )

    hloc = (0.25*local_parity_operator @ ((np.cos(delta)*(np.cos(phi)*np.kron(sX,sX) + np.sin(phi)*np.kron(sY,sY)) - np.sin(delta)*np.kron(sZ,sZ))) @ local_parity_operator).reshape(2,2,2,2)

    return hloc


#########################  THE REST OF THE FUNCTIONS USED IN THE CODES ####################

def doApplyHamTENSOR(psi: np.ndarray, h: np.ndarray, N: int, usePBC: bool):
    """ Basic succesive application of the local 4 leg local Hamiltonian to the psi tensor. 
        This routine is designed to work with the LinearOperator function and the Lanczos type algorithms.
        Args:
            psi:        vector of length d**N describing the quantum state.
            h:          array of ndim=4 describing the nearest neighbor coupling (local Hamiltonian of dimension (d^2, d^2)).
            N:          the number of lattice sites.
            usePBC:     sets whether to include periodic boundary term.
        Returns:
            np.ndarray: state psi after application of the Hamiltonian.

        Logic of the application is explained somewhere in the notes. 
        First, second and last application of the local Hamiltonian are written explicelty. The remaining steps are generic and the same algorithm is applied.      
    """
    # this is the local dimension 
    d = 2

    # reshape the vector into a tensor with N legs each with dimension d
    reshape_array_default     = np.full(N,d)
    psi = psi.reshape(reshape_array_default)

    # First application of the local Hamiltonian to the tensor quantum state for legs 1 and 2 (k = 0)
    psiOut = (h.reshape(4,4) @ psi.reshape(4,2**(N-2))).reshape(reshape_array_default).reshape(2**N)


    # Second application of the local Hamiltonian to the tensor quantum state for legs 2 and 3 (k = 1)
    # generate the lists of transposition (first transpose different from the second one)
    transpose_array_default1    = np.arange(N)
    transpose_array_default2    = transpose_array_default1.copy()

    transpose_array_default1[0] = 1
    transpose_array_default1[1] = 2
    transpose_array_default1[2] = 0 

    transpose_array_default2[0] = 2
    transpose_array_default2[1] = 0
    transpose_array_default2[2] = 1 

    psiOut += (h.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    # Remaining applications of the local Hamiltonian  (k >= 2)
    for k in range(2,N-1):

        transpose_array_default         = np.arange(N)
        transpose_array_default[0]      = k
        transpose_array_default[1]      = k+1
        transpose_array_default[k]      = 0 
        transpose_array_default[k+1]    = 1

        psiOut += (h.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

        
    # PBC conditions application of the local hamiltonian on the end tails of the psi tensor
    if usePBC:

        # generate the lists of transposition (first transpose different from the second one)
        transpose_array_default1  = np.arange(N)
        transpose_array_default2  = transpose_array_default1.copy()

        transpose_array_default1[0]    = N - 1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psiOut += (h.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    return psiOut


def local_magnetization_single_point(sigma,j,psi,N,d):
    """ function to evaluate single point correlator < psi | sigma_{j}^{x,y,z} | psi > - magnetization """ 

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default     = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default        = np.arange(N)
    transpose_array_default[0]     = j-1
    transpose_array_default[j-1]   = 0

    psi = (sigma @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = (np.transpose(np.conjugate(initial_psi))) @ psi

    return mag  

def local_magnetization_two_point(sigma,j,t,psi,N,d):
    """ # function to evaluate two point spin correlator < psi | sigma_{j + t}^{x,y,z} sigma_{j}^{x,y,z} | psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = j-1
    transpose_array_default[j-1]  = 0

    # first operator application
    psi = (sigma @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = (j+t-1) % N               
    transpose_array_default[(j+t-1) % N]  = 0


    # second operator application
    psi = (sigma @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  


def local_magnetization_two_point_general(sigma2,sigma1,j,t,psi,N,d):
    """ # function to evaluate two point spin correlator < psi | sigma_{j + t}^{x,y,z} sigma_{j}^{x,y,z} | psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = j-1
    transpose_array_default[j-1]  = 0

    # first operator application
    psi = (sigma1 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = t-1               
    transpose_array_default[t-1]  = 0


    # second operator application
    psi = (sigma2 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  

def local_magnetization_general_point_general(sigma,jlist,psi,N,d):
    """ # function to evaluate two point spin correlator < psi | sigma_{j1}^{x,y,z} sigma_{j2}^{x,y,z} ... sigma_{jN}^{x,y,z}| psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # general loop
    for ii in range(0,len(jlist)):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = jlist[ii]-1
        transpose_array_default[jlist[ii]-1]  = 0

        # first operator application
        psi = (sigma[ii] @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    psi = psi.reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  


def local_magnetization_two_neighboring_point(sigma,j,psi,N,d):
    """ this is the function which computes < psi | sigma_{j}^{x,y,z} sigma_{j+1}^{x,y,z} | psi > - this should give the same output as the function :local_magnetization_two_point: for t = 1   """

    sigma = np.kron(sigma, sigma)

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)


    if (j == N):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = N-1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)

    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1


        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi


    return mag

def tfim_ground_state_energy_per_site_exact(N,J,h):
    """
    Exact results in Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 """

    integral = quad(tfim_ground_state_energy_per_site_exact_integrand, 0, np.pi, args=(J, h))[0]
    return -(h/(2.0*np.pi))*integral

def tfim_ground_state_energy_per_site_exact_integrand(q, J, h):
    return (cmath.sqrt(1.0 + (J**2)/(4*h**2) - (J*cmath.cos(q))/(h) )).real


def tfim_ground_state_magnetizationSigmaZ_per_site_exact(N,J,h):
    """
    Exact results in Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 """

    integral = quad(tfim_ground_state_magnetization_per_site_exact_integrand, 0, np.pi, args=(J, h))[0]
    return (1.0/(np.pi))*integral

def tfim_ground_state_magnetization_per_site_exact_integrand(q, J, h):

    return ((1.0 - (J*cmath.cos(q))/(2.0*h))/((cmath.sqrt(1.0 + (J**2)/(4*h**2) - (J*cmath.cos(q))/(h) )))).real


def tfim_ground_state_magnetizationSigmaX_per_site_exact(N,J,h):
    """
    Exact results in Analysis of the transverse field Ising model by continuous unitary transformations. 
    Master Thesis, Lehrstuhl für Theoretische Physik I
    Fakultät Physik
    Technische Universität Dortmund
    Benedikt Fauseweh
    2012 """

    if (J >= 2.0*h):
        value = (1.0 - (4.0*h**2)/(J**2))**(1.0/8.0)

    if (J < 2.0*h):
        value = 0.0


    return value


def PartialTraceGeneralTensor(N,index_list,A):
    """ Function that computes the partial trace over index_list indices (the index list needs to be ordered from smaller to bigger index)"""

    # reshape the input vectors into tensors (here we exploit the fact that psi* is just the complex conjugate of psi )
    reshape_array_default = np.full(N,2)    
    A_initial = A.reshape(reshape_array_default)

    # generate initial transpose indices vector (we apply permutations and operatorion so transposition is correctly performed )
    list_A = np.arange(N)
    list_B = np.arange(N)

    # this changing the indeces by one is because of python stuff (the numbering starts from zero and not 1)
    index_list = np.array(index_list) - 1

    ##### generating the first transpose rule for A ###

    ## initial step of moving the indices to the left
    for zz in range(0,len(index_list)):
        list_A[zz] = index_list[zz]

    ## figure out what are the missing indices that happen because of overwritting in loop above
    list_A_no_dupl = list(OrderedDict.fromkeys(list_A))
    missing_indices = np.delete(np.arange(N), list_A_no_dupl)

    ## now replace the doubled indices with indices in the missing_indices array
    counter = 0
    for zz in range(len(index_list),len(list_A)):
        for tt in range(0,len(index_list)):
            if (list_A[zz] == index_list[tt]):
                list_A[zz] = missing_indices[counter]
                counter += 1
    

    ##### generating the first transpose rule for B ###

    ## initial step of moving the indices to the right
    for zz in range(0,len(index_list)):
        list_B[len(list_B) - zz - 1] = index_list[len(index_list) - zz - 1]

    ## figure out what are the missing indices that happen because of overwritting in loop above
    list_B_no_dupl = list(OrderedDict.fromkeys(list_B))
    missing_indices = np.delete(np.arange(N), list_B_no_dupl)

    ## now replace the doubled indices with indices in the missing_indices array
    counter = 0
    for zz in range(0,len(list_B)-len(index_list)):
        for tt in range(0,len(index_list)):
            if (list_B[zz] == index_list[tt]):
                list_B[zz] = missing_indices[counter]
                counter += 1


    ##### generating the second transpose rule for A ###

    list_A_cut = list_A[len(index_list):]
    list_A_cut_sort = np.sort(list_A_cut)

    list_B_cut = list_B[:-len(index_list)]
    list_B_cut_sort = np.sort(list_B_cut)

    transpose2_A = np.append(index_list,list_A_cut_sort)
    transpose2_B = np.append(list_B_cut_sort,index_list)

    ############### MAIN OPERATION AFTER ALL PREPARATION HAS BEEN PERFORMED ::: TRANSPOSITION ON A and B
    A = A_initial.transpose(transpose2_A).reshape(2**len(index_list),2**(N - len(index_list)))
    B = A_initial.transpose(transpose2_B).reshape(2**(N - len(index_list)),2**len(index_list))

    # FINAL MULTIPLICATION
    out = (A @ np.conjugate(B))

    return out


def PauliMatrice():

    """ Defining the Pauli matrices """

    sX = np.array([[0, 1.0], [1.0, 0]])
    sY = np.array([[0, -1.0j], [1.0j, 0]])
    sZ = np.array([[1.0, 0], [0, -1.0]])
    sI = np.array([[1.0, 0], [0, 1.0]])

    return sX,sY,sZ,sI

def PartialTraceGeneralCorrelation(N,index_list,A,B):
    """ Evalute the Partial Trace by evaluating spin point correlations function. This was used as a check. It turns out that evalauting concurrence is much faster anyway using direct partial trace when in tensor representation. """
    
    reverse_index_list = index_list[::-1]

    # load in the Pauli Matrices
    sI,sX,sY,sZ = PauliMatrice()

    # generate 
    x = [sI,sX,sY,sZ]
    list_of_2point_correlatores = [p for p in itertools.product(x, repeat=len(index_list))]

    value = np.zeros((2,2),dtype=complex)
    for ii in range(0,len(index_list)-1):
        value = np.kron(value, np.zeros((2,2),dtype=complex) )


    for i in range(0,len(list_of_2point_correlatores)):


        # generate the list of operators that needs to be evalued
        sigmalist = []
        for ii in range(0,len(index_list)):
            sigmalist.append(list_of_2point_correlatores[i][ii])

        # generate the tensor product of those same operators in the list above
        kkron = list_of_2point_correlatores[i][0]
        for ii in range(1,len(index_list)):
                kkron = np.kron(kkron,list_of_2point_correlatores[i][ii] )


        value = value + local_magnetization_general_point_general(sigmalist,reverse_index_list,psi,N,2)*kkron


    return value

def local_magnetization_two_point_general(sigma2,sigma1,j,t,psi,N,d):
    """  function to evaluate two point spin correlator < psi | sigma_{j + t}^{x,y,z} sigma_{j}^{x,y,z} | psi > - ONLY MADE TO WORK WITH PBC IMPOSED ON THE PROBLEM (see later mod function % which loops around psi) """

    initial_psi = psi.copy()

    # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = j-1
    transpose_array_default[j-1]  = 0

    # first operator application
    psi = (sigma1 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default)

    # transpose the tensor with N legs to match the position of the local operator
    transpose_array_default       = np.arange(N)
    transpose_array_default[0]    = t-1               
    transpose_array_default[t-1]  = 0


    # second operator application
    psi = (sigma2 @ psi.transpose(transpose_array_default).reshape(d,2**(N-1))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    mag = np.transpose(initial_psi) @ psi

    return mag  



def ApplyLocalGate(sigma,index_pair,psi,N,d,dt):
    """ apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC"""

    sigma = linalg.expm(-1j*dt*sigma)

    j = index_pair[0]

   # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    if (j == N):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = N-1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)


    return psi

def ApplyLocalGate_GENERAL(sigma,index_pair,psi,N,d,dt):
    """ apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC. THIS IS GENERAL BECAUSE WE DO NOT EXPONENTIATE THE HERMITIAN MATRICES INSIDE"""


    j = index_pair[0]

   # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default         = np.full(N,d)    
    psi = psi.reshape(reshape_array_default)

    if (j == N):
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = N-1
        transpose_array_default1[1]    = 0
        transpose_array_default1[N-1]  = 1

        transpose_array_default2[0]    = 1
        transpose_array_default2[1]    = N-1
        transpose_array_default2[N-1]  = 0

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)


    elif(j == 2):

        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default1       = np.arange(N)
        transpose_array_default2       = transpose_array_default1.copy()

        transpose_array_default1[0]    = 1
        transpose_array_default1[1]    = 2
        transpose_array_default1[2]    = 0

        transpose_array_default2[0]    = 2
        transpose_array_default2[1]    = 0
        transpose_array_default2[2]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1

        psi = (sigma.reshape(4,4) @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)


    return psi



def concurrence_measure(rho):
    """ computing the concurrence measure of entanglement """

    sX, sY, sZ, sI = PauliMatrice()

    rho_sqrt = sqrtm(rho)

    rho_fin =  sqrtm( rho_sqrt @ (np.kron(sY,sY) @ np.conjugate(rho) @ np.kron(sY,sY)) @ rho_sqrt )


    eigenvalues,eigenvectors = LA.eigh(rho_fin)
    idx = eigenvalues.argsort()[::-1]   
    eigenvalues = eigenvalues[idx]

    con =  np.max([eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3], 0.0])


    return con


def generate_dictionary(N :int,r :int, usePBC :bool):
    """ generate the list of pairs for the application of the local Hamiltonian  """
    lista = []

    for iii in range(0,N-r):
        lista.append([iii+1,iii+1+r])


    if usePBC:
        for iii in range(0,r):
            lista.append([N-iii,r-iii])



    return lista



def magnetizationSigmaZ_per_site_exact(lambdaa):
    """
    
    magnetization exact result

    PHYSICAL REVIEW B 78, 224413 共2008兲


    """

    integral = quad(ground_state_magnetization_per_site_exact_integrand, 0, np.pi, args=(lambdaa))[0]
    return -(1.0/(np.pi))*integral

def ground_state_magnetization_per_site_exact_integrand(q, lambdaa):
    """" magnetizaion exact result integrand under the integral """
    return ( (1.0 + lambdaa*np.cos(q) )/(np.sqrt( (lambdaa*np.sin(q))**2.0 + (1.0 + lambdaa*np.cos(q))**2.0 ) ) ).real


def select_cooling_evolution_gates():
    """ defines a set of evolution gates from which we choose one random uniformly with an equal probability """
    # define Pauli matrices
    sX, sY, sZ, sI = PauliMatrice()

    # define the 3 'single' and 3 two body operators
    single_body1 =  np.kron(sX,sI) + np.kron(sI,sX)
    single_body2 =  np.kron(sY,sI) + np.kron(sI,sY)
    single_body3 =  np.kron(sZ,sI) + np.kron(sI,sZ)

    two_body1 = np.kron(sX,sX)
    two_body2 = np.kron(sY,sY)
    two_body3 = np.kron(sZ,sZ)

    gate_list = [single_body1,single_body2,single_body3,two_body1,two_body2,two_body3]

    # select a random values (uniform)
    rand_value = np.random.uniform(0,1,1)

    idd = 0
    # take the gate from the gate_list with equal probability
    for ij in range(0,6):
        if (rand_value <= (1.0/6.0)*(ij + 1)):
            if (rand_value > (1.0/6.0)*(ij)):
                gate = gate_list[ij]
                idd = ij + 1



    return gate,idd



def select_cooling_evolution_gates_GENERAL(dt):
    """ defines a set of evolution gates from which we choose one random uniformly with an equal probability """
    # define Pauli matrices
    sX, sY, sZ, sI = PauliMatrice()


    two_body1 = np.kron(sX,sX)
    two_body2 = np.kron(sY,sY)
    two_body3 = np.kron(sZ,sZ)

    gate_list = [two_body1,two_body2,two_body3]

    # select a random values (uniform)
    rand_value = np.random.uniform(0,1,1)

    idd = 0
    # take the gate from the gate_list with equal probability
    for ij in range(0,3):
        if (rand_value <= (1.0/3.0)*(ij + 1)):
            if (rand_value > (1.0/3.0)*(ij)):
                gate = gate_list[ij]
                idd = ij + 1

    gate = linalg.expm(-1j*dt*gate)

    return gate,idd

def select_cooling_evolution_indices(list_pairs):
    """ select a random pair of indices (adjacent) to which we apply to the local gates later """

    # select a random value (uniform)
    rand_value = np.random.uniform(0,1,1)

    # take the random pair with equal probability
    for ij in range(0,len(list_pairs)):
        if (rand_value <= (1.0/len(list_pairs))*(ij + 1)):
            if (rand_value > (1.0/len(list_pairs))*(ij)):
                pair = list_pairs[ij]

    # find where the randomly selected pair is located in the initial list of pairs
    idd =  list_pairs.index(pair)


    return pair, idd


def concurrence_from_state(Nsites,psi,whichone):
    """ compute the concurrence from a given state at distance r """
    r = whichone

   # generate the list of pairs for concurence measurement
    list_pairs = generate_dictionary(Nsites,r, True)

    # compute the average concurrence
    conn11 = []
    for jj in range(0,len(list_pairs)):

        rho = PartialTraceGeneralTensor(Nsites,list_pairs[jj],psi) 

        conn11.append(concurrence_measure(rho))

    concurrence = np.average(conn11)


    return concurrence

def concurrence_from_state_summ(Nsites,psi,upto):
    """ compute the concurrence from a givn state, but compute and add up nearest, next, next-to... nearest neighbors (r = 1, 2, 3,... upto) into a single number for each of the possible pairings at each level"""

    # loop over the r's
    add = 0
    for kk in range(1,upto + 1 ):
        # generate the list of pairs for concurence measurement
        list_pairs = generate_dictionary(Nsites,kk, True)

        # compute the average concurrence
        conn11 = []
        for jj in range(0,len(list_pairs)):
            rho = PartialTraceGeneralTensor(Nsites,list_pairs[jj],psi) 

            conn11.append(concurrence_measure(rho))


        add = add + np.average(conn11)

    concurrence = add


    return concurrence



def vonNeumann(N,R,psi):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

#        print("entropy------------------------------------------")
 #       print(entropy)
        ee.append(entropy)        


    entropy = - np.average(ee)


    return entropy,ee

def vonNeumann_natbase(N,R,psi):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log(eigenvalues[ii])

#        print("entropy------------------------------------------")
 #       print(entropy)
        ee.append(entropy)        


    entropy = - np.average(ee)


    return entropy,ee


def vonNeumann_aftergate(N,R,psi,gate_id):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)


    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):
        
        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])

        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])    




    ee = []
    for i in range(0,len(kk)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)


        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

        ee.append(entropy)        

    entropy = - np.average(ee)


    return entropy,ee


def vonNeumann_aftergate_correct(N,R,psi,gate_id):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)


    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):
        
        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])
            #print(list_indic[i],i)
            relevant_list.append(i)
        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])    
            #print(list_indic[i],i)
            relevant_list.append(i)



    ee = []
    for i in range(0,len(kk)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        #print('hahaa = ',len(eigenvalues))


        for ii in range(0,len(eigenvalues)):

            if (eigenvalues[ii] > epsi):
                #print(ii,eigenvalues[ii])
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])

        ee.append(entropy)        

    entropy = - np.average(ee)


    return entropy,ee,relevant_list


def vonNeumann_aftergate_correct_natbase(N,R,psi,gate_id):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)


    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):
        
        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])
            #print(list_indic[i],i)
            relevant_list.append(i)
        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])    
            #print(list_indic[i],i)
            relevant_list.append(i)



    ee = []
    for i in range(0,len(kk)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        #print('hahaa = ',len(eigenvalues))


        for ii in range(0,len(eigenvalues)):

            if (eigenvalues[ii] > epsi):
                #print(ii,eigenvalues[ii])
                entropy += eigenvalues[ii]*np.log(eigenvalues[ii])

        ee.append(entropy)        

    entropy = - np.average(ee)


    return entropy,ee,relevant_list

def generate_dictionary_adjacent(N :int, r :int):
    """ generate the list used evaluate of the von Neuman entropy, and only adjacent spins are to be considered  """

    lista = []
    r = r - 1 
    for iii in range(0,N):
        ll_tampon = [(iii ) % (N)]
        for jjj in range(0,r):
            ll_tampon.append((iii+jjj+1) % (N))
    
        for ii in range(0,len(ll_tampon)):
            ll_tampon[ii] += 1 
        lista.append(ll_tampon)


    return lista    


def func_for_fit(x, a, b, c):
    """ Fitting function for the concurrence in the odd case. """ 
    return a*np.array(x)**b  + c

def func_for_fit_old(x, a, b):
    """ Fitting function for the concurrence in the odd case. """ 
    return a*np.array(x)**b




def U(dt :float):
    """ Generating 2x2 random unitary matrix (from 1. Zyczkowski K, Kus M. Random unitary matrices. J Phys A: Math Gen [Internet]. 1994 Jun [cited 2021 Mar 26];27(12):4235–45. ) """ 
    alpha, psi, chi = np.random.uniform(low=0.0, high=2*np.pi, size=3)
    phi = np.arcsin(np.sqrt(np.random.uniform(low=0.0, high=1.0)))

    res = np.zeros((2,2),dtype=np.complex64)
    res[0,0] = np.exp(1.0j*psi)*np.cos(phi)
    res[0,1] = np.exp(1.0j*chi)*np.sin(phi)
    res[1,0] = -np.exp(-1.0j*chi)*np.sin(phi)
    res[1,1] = np.exp(-1.0j*psi)*np.cos(phi)

    return np.exp(1.0j*alpha) * res



def random_gate(dt :float):
    """ the total 4x4 sized random unitary actually is constructed from a tensor product of two 2x2 tensors multiplied with a tensor of 4x4 which is constructed directly from the three nontricial gates sigma_x sigma_x, sigma_y sigma_y, sigma_z sigma_z"""
    U1 = U(dt)
    U2 = U(dt)

    ridx = np.random.randint(3)

    sX, sY, sZ, sI = PauliMatrice()


    if (ridx == 0):
        Uxx = linalg.expm(1.0j*dt*np.kron(sX,sX))
        return np.kron(U1, U2) @ Uxx, ridx
    elif (ridx == 1):
        Uyy = linalg.expm(1.0j*dt*np.kron(sY,sY))
        return np.kron(U1, U2) @ Uyy, ridx
    else:
        Uzz = linalg.expm(1.0j*dt*np.kron(sZ,sZ))
        return np.kron(U1, U2) @ Uzz, ridx

def wignersurmiseGUE(x):
	""" Wigner surmise for GUE - just expression"""
	return ((32.0*x*x)/(np.pi*np.pi))*np.exp(-(4.0/np.pi)*x*x)



def histogram_generate(bins,array):						
	""" generating a line from a histogram data, so given a 1d array of data this function creates a line plot data """
	minarray = np.min(array)
	maxarray = np.max(array)
	totalsize = abs(minarray) + abs(maxarray)
	binsize = totalsize/bins
	
	
	# defining the histogram array
	histogramcounter = []
	for i in range(0,bins):
		histogramcounter.append(0)

	# filling in the histogram array
	for i in range(0,len(array)):
		#print(i)
		for j in range(0,bins):
			if (array[i] >= minarray + j*binsize):
				if (array[i] <= minarray + (j+1)*binsize):
					histogramcounter[j]= histogramcounter[j] + 1

	# due to bad coding I add the largest value manually (dumb but works)
	histogramcounter[bins-1] = histogramcounter[bins-1] + 1
	
	# generate X axis of the histogram
	histoX = []
	for i in range(0,bins):
		histoX.append(minarray + i*binsize + binsize/2.0 )

	# normalization
	histoNORM = []
	norm = 0.0
	for i in range(0,bins):
		histoNORM.append((bins/totalsize)*(histogramcounter[i]/len(array)))
		norm = norm + (bins/totalsize)*(histogramcounter[i]/len(array))*binsize
	
	return histogramcounter,binsize, histoX, histoNORM			# OUTPUT: unnormed histogram, bins size, histogram X azis, normed histogram Y axis


def Maximally_entangled_Bell_state(N: int, theta: float):

    zero = np.array([0.0,1.0])
    one  = np.array([1.0,0.0])


    zero_iterate = zero
    one_iterate  = one
    for i in range(0,N-1):
        zero_iterate = np.kron(zero_iterate,zero)
        one_iterate  = np.kron(one_iterate, one)

    psi = np.cos(theta)*zero_iterate  + np.sin(theta)*one_iterate

    return psi

def vonNeumann_test(N,R,psi):
    """ compute the von Neumann entropy of a state psi with the subsystem being size R  """

    # fix the R 
    R = int(math.floor(R))
    #R = int(math.ceil(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):
        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        # compute eigenvalues
        eigenvalues,eigenvectors = LA.eigh(rho_reduced)
        idx = eigenvalues.argsort()[::-1]   
        eigenvalues = eigenvalues[idx]

        # compute the von Neumann entropy
        entropy = 0.0
        epsi = 10**(-15)

        print('hahaa = ',len(eigenvalues))

        for ii in range(0,len(eigenvalues)):
            if (eigenvalues[ii] > epsi):
                entropy += eigenvalues[ii]*np.log2(eigenvalues[ii])
               # print(ii,eigenvalues[ii])

#        print("entropy------------------------------------------")
 #       print(entropy)
        ee.append(entropy)        


    entropy = - np.average(ee)


    return entropy,ee


""" NOT USED FUNCTIONS, we need to decied what do with them and if we need them in this form


def createGates(path, numGates):

    for i in range(numGates):
        aa, ii = gate(0.3)
        np.savetxt(path + "gate_" + str(i+1) + "_real.csv", np.real(aa),  delimiter=",")
        np.savetxt(path + "gate_" + str(i+1) + "_img.csv", np.imag(aa),  delimiter=",")


def createIndexes(path, numIndexes):
    idxlist = []
    for i in range(numIndexes):
        idxlist.append(np.random.randint(1,7))

    np.savetxt(path + "randomidx_" + str(numIndexes) + ".csv", idxlist,  delimiter=",")


def createRandomProbabilities(path, numProbs):
     ll = np.random.uniform(low=0.0, high=1.0, size=numProbs)
     np.savetxt(path + "probabily_list_" + str(numProbs) + ".csv", ll,  delimiter=",")
"""