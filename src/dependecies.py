# needed libraries
import numpy as np
import cupy as cp
from mpi4py import MPI

from scipy.integrate import quad
import cmath
from collections import OrderedDict
import itertools
from scipy import linalg
from numpy import linalg as LA
from scipy.linalg import sqrtm
import math
import time
import nvtx
import socket


class Timing:
    __dict = {"TotalTime": [],
              "ApplyLocalGate" : [],
              "RenyEntropy": [],
              "PartialTrace": []}

    def set_mode(self, mode, timeit):
        self.mode = mode
        self.timeit  = timeit


    def Start(self, timeName):
        if (self.timeit):
            if (not self.mode == "CPU"):
                start_gpu = cp.cuda.Event()
                start_gpu.record()
                self.__dict[timeName].append(start_gpu)
                
            else:
                self.__dict[timeName].append(MPI.Wtime())


    def Stop(self, timeName):
        if(self.timeit):
            if (not self.mode == "CPU"):
                end_gpu = cp.cuda.Event()
                end_gpu.record()
                end_gpu.synchronize()
                self.__dict[timeName][-1] = cp.cuda.get_elapsed_time(self.__dict[timeName][-1], end_gpu)/1000
            else:
                self.__dict[timeName][-1] = MPI.Wtime()- self.__dict[timeName][-1]

    def __str__(self):
        for key in self.__dict.keys():
            self.__dict[key] = np.sum(self.__dict[key])

        if(self.timeit):
            return (f'\n{"Timing:":-^30}\n'
                    f'|{"total time:":<20s}{self.__dict["TotalTime"]:8.2f}|\n'
                    f'|{"apply local gate:":<20s}{self.__dict["ApplyLocalGate"]:8.2f}|\n'
                    f'|{"renyi2 entropy:":<20s}{self.__dict["RenyEntropy"]:8.2f}|\n'
                    f'|{"partial trace:":<20s}{self.__dict["PartialTrace"]:8.2f}|\n'
                    f'{"":-^30}')
        else:
            return "Timing not on"

#### GLOBALS #####
timing  = Timing()
Streams = []
##################

def tfim_LocalHamiltonian_new(lambdaa):

    """ Trasverse Ising Hamiltonian - Fabio/Marco """ 

    sX, sY, sZ, sI = PauliMatrice()

    hloc = (lambdaa*np.kron(sX, sX) - 0.5*(np.kron(sZ,sI) + np.kron(sI,sZ)) ).reshape(2, 2, 2, 2)
    #hloc = (lambdaa*np.kron(sX, sX) - np.kron(sZ,sI) ) .reshape(2, 2, 2, 2)

    return hloc  
#########################  THE REST OF THE FUNCTIONS USED IN THE CODES ####################
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

def tfim_LocalHamiltonian_new(lambdaa):

    """ Trasverse Ising Hamiltonian - Fabio/Marco """ 

    sX, sY, sZ, sI = PauliMatrice()
    hloc = (lambdaa*np.kron(sX, sX) - 0.5*(np.kron(sZ,sI) + np.kron(sI,sZ)) ).reshape(2, 2, 2, 2)
    #hloc = (lambdaa*np.kron(sX, sX) - np.kron(sZ,sI) ) .reshape(2, 2, 2, 2)

    return hloc 

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
    timing.Start("PartialTrace")
    out = (A @ np.conjugate(B))
    timing.Stop("PartialTrace")

    return out

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

def PauliMatrice():

    """ Defining the Pauli matrices """

    sX = np.array([[0, 1.0], [1.0, 0]])
    sY = np.array([[0, -1.0j], [1.0j, 0]])
    sZ = np.array([[1.0, 0], [0, -1.0]])
    sI = np.array([[1.0, 0], [0, 1.0]])

    return sX,sY,sZ,sI

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

def generate_dictionary(N :int,r :int, usePBC :bool):
    """ generate the list of pairs for the application of the local Hamiltonian  """
    lista = []

    for iii in range(0,N-r):
        lista.append([iii+1,iii+1+r])


    if usePBC:
        for iii in range(0,r):
            lista.append([N-iii,r-iii])



    return lista

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

def Renyi2(N,R,psi):
    """ compute the Renyi entropy alpha = 2 (easy case)  of a state psi with the subsystem being size R  """


    # fix the R 
    R = int(math.floor(R))
   
    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    ee = []
    for i in range(0,len(list_indic)):

        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,list_indic[i],psi) 

        entropy = -np.log((rho_reduced * rho_reduced.T).sum())
        ee.append(entropy)        

    entropy = np.average(ee)

    return entropy,ee

def Renyi2_aftergate_correct(N,R,psi,gate_id):
    """ compute the Renyi 2 entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R
    R = int(math.floor(R))

    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):

        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])
            relevant_list.append(i)
        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])
            relevant_list.append(i)

    ee = []
    for i in range(0,len(kk)):


        # compute the partial trace
        rho_reduced = PartialTraceGeneralTensor(N,kk[i],psi)
	# compute the trace of the product of two reduceded density matrices
        entropy = -np.log((rho_reduced * rho_reduced.T).sum())
        
        ee.append(entropy.real)
		
    entropy = np.average(ee)

    return entropy,ee,relevant_list

def set_streams_global( batch_size):
    global Streams

    for _ in range(batch_size):
        Streams.append( cp.cuda.Stream())


def create_sigma_list_GPU(dt):

    """ create a set of evolution gates in CUDA memory, accessed by GLOBAL list named gate_list_DEVICE"""
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
    
    sigma_list = []
    for i in range(len(gate_list)):
        sigma_list.append( linalg.expm(-1j*dt*gate_list[i]) )

    #global sigma_list_DEVICE 
    sigma_list_DEVICE = cp.asarray(sigma_list)
    return sigma_list_DEVICE


@nvtx.annotate("ApplyLocalGate", color="blue")
def ApplyLocalGate_GPU(sigma_DEVICE,index_pair,psi,N,d,dt):
    """ apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC"""

    #print(cumulative_time_gemm_apply_lg)
    j = index_pair[0]

   # reshape into a tensor with N legs, each leg with dimension d
    reshape_array_default        = np.full(N,d)    
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

        #psi = (sigma_DEVICE @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
        
        # Prepare tensor for gemm operation      
        psi = psi.transpose(transpose_array_default1).reshape(4,2**(N-2))

        psi = sigma_DEVICE @ psi

        psi = psi.reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)

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

        #psi = (sigma_DEVICE @ psi.transpose(transpose_array_default1).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)

        # Prepare tensor for gemm operation        
        psi = psi.transpose(transpose_array_default1).reshape(4,2**(N-2))

        # Gemm operation 
        psi = sigma_DEVICE @ psi

        # Transpose and rashape bach to origininal shape tensor
        psi = psi.reshape(reshape_array_default).transpose(transpose_array_default2).reshape(2**N)
      
    else:    
        # transpose the tensor with N legs to match the position of the local operator
        transpose_array_default       = np.arange(N)
        transpose_array_default[0]    = j-1
        transpose_array_default[1]    = j
        transpose_array_default[j-1]  = 0
        transpose_array_default[j]    = 1

        #psi = (sigma_DEVICE @ psi.transpose(transpose_array_default).reshape(4,2**(N-2))).reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

        # Prepare tensor for gemm operation        
        psi = psi.transpose(transpose_array_default).reshape(4,2**(N-2))

        # Gemm operation 
        psi = sigma_DEVICE @ psi

        # Transpose and rashape bach to origininal shape tensor

        psi = psi.reshape(reshape_array_default).transpose(transpose_array_default).reshape(2**N)

    return psi

@nvtx.annotate("PartialTrace" , color="black")
def PartialTraceGeneralTensor_new_GPU(N,index_list, A):
    """ Function that computes the partial trace over index_list indices (the index list needs to be ordered from smaller to bigger index)"""

    global cumulative_time_gemm_partial_trace
    global counter_gemm_partial_trace

    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    # reshape the input vectors into tensors (here we exploit the fact that psi* is just the complex conjugate of psi )
    reshape_array_default = np.full(N,2)    
    A_initial = A.reshape(reshape_array_default)

    # generate initial transpose indices vector (we apply permutations and operatorion so transposition is correctly performed )
    list_A = np.arange(N)
    list_B = np.arange(N)

    # this changing the indeces by one is because of python stuff (the numbering starts from zero and not 1)
    index_list = np.array(index_list) - 1
    
    A_temp = np.take(list_A, index_list)
    B_temp = np.take(list_B, index_list)

    transpose2_A = np.append(A_temp, np.delete(list_A, index_list) )
    transpose2_B = np.append(np.delete(list_B, index_list), B_temp )

    ############### MAIN OPERATION AFTER ALL PREPARATION HAS BEEN PERFORMED ::: TRANSPOSITION ON A and B
    A_DEVICE = A_initial.transpose(transpose2_A).reshape(2**len(index_list),2**(N - len(index_list)))
    B_DEVICE = A_initial.transpose(transpose2_B).reshape(2**(N - len(index_list)),2**len(index_list))

    # FINAL MULTIPLICATION
    timing.Start("PartialTrace")
    out_DEVICE = (A_DEVICE @ cp.conjugate(B_DEVICE))
    timing.Stop("PartialTrace")
    
    return (out_DEVICE)

@nvtx.annotate("Renyi2", color="purple")
def Renyi2_aftergate_correct_GPU(N,R,psi_DEVICE,gate_id):
    """ compute the Renyi 2 entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R
    R = int(math.floor(R))

    # generating the indices that get traced out
    list_indic = generate_dictionary_adjacent(N,R)

    kk = []
    relevant_list = []
    for i in range(0,len(list_indic)):

        if (list_indic[i][0] == gate_id[1]):
            kk.append(list_indic[i])
            relevant_list.append(i)
        if (list_indic[i][R-1] == gate_id[0]):
            kk.append(list_indic[i])
            relevant_list.append(i)

    ee_DEVICE = []
    psi_DEVICE2 = psi_DEVICE.copy()
    psis_DEVICE = [psi_DEVICE, psi_DEVICE2] 
    rho_reduced_list = []
    for i in range(0,len(kk)):

        # compute the partial trace
        rho_reduced_list.append( PartialTraceGeneralTensor_new_GPU(N,kk[i],psis_DEVICE[i]))
	    
        #compute the trace of the product of two reduceded density matrices
        ee_DEVICE.append((-cp.log((rho_reduced_list[i] * rho_reduced_list[i].T).sum())).real )

        
    ee_DEVICE = cp.asarray(ee_DEVICE)

    entropy_DEVICE = cp.average(ee_DEVICE)

    return entropy_DEVICE,ee_DEVICE,relevant_list


def select_cooling_evolution_indices_batch(list_pairs, batch_size):
    """ select a random pair of indices (adjacent) to which we apply to the local gates later """
    pair_list = []
    idd_list  = []
    
    for _ in range(batch_size):

        # select a random value (uniform)
        rand_value = np.random.uniform(0,1,1)

        # take the random pair with equal probability
        for ij in range(0,len(list_pairs)):
            if (rand_value <= (1.0/len(list_pairs))*(ij + 1)):
                if (rand_value > (1.0/len(list_pairs))*(ij)):
                    pair = list_pairs[ij]

        # find where the randomly selected pair is located in the initial list of pairs
        idd =  list_pairs.index(pair)
        pair_list.append(pair)
        idd_list.append(idd)


    #return pair, idd
    return pair_list, idd_list


def select_sigma_gates_batch(sigmas, batch_size):
	
	sigma_gates = []
	sigma_random_select_batch_list = np.random.randint(0, high = len(sigmas) , size = batch_size)

	for i in range(batch_size):
	
		sigma_gates.append( sigmas[ sigma_random_select_batch_list[i]])
		
	return cp.array(sigma_gates)

@nvtx.annotate("ApplyLocalGate_batch", color="blue")
def ApplyLocalGate_GPU_batch(sigmas, batch_size, index_pair, psi, N, d, dt):
    """ apply exponentiated local 2 site gate, this is generalized to any distance pair but does not involve pairs that are there due to PBC"""

    tmp_psi = []
    transpose_array_default_batch = []

    #set global variables 
    global cumulative_time_gemm_apply_lg
    global cumulative_time_prepare_apply_lg
    global counter_gemm_local_gate
    global Streams

    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()

    # Create reshape array for whole batched state 
    # np.array(batch_size, 2, 2, .... N)   
    reshape_array_default_batch = np.insert(np.full(N, d), 0, batch_size)
    
    #with nvtx.annotate("Apply_local_gate_batched_reshape", color="purple"):
    
    psi = psi.reshape( reshape_array_default_batch)
    
    for batch_num in range(batch_size):
        j = index_pair[batch_num][0]

        # Set streams for possible overlape of reshape and 
        # transpose function on GPU
        Streams[batch_num].use()

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

            # for now psi[batch_num].operation... instead of tmp_psi[batch_num].operation
            start_gpu.record()

            tmp_psi.append( psi[batch_num].transpose(transpose_array_default1).reshape(4,2**(N-2)) )
            with nvtx.annotate("psi_1", color="purple"):
                transpose_array_default_batch.append(transpose_array_default2)
            
            end_gpu.record()
            end_gpu.synchronize()
            cumulative_time_prepare_apply_lg += cp.cuda.get_elapsed_time(start_gpu, end_gpu)

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
            
            # for now psi[batch_num].operation... instead of tmp_psi[batch_num].operation
            start_gpu.record()

            with nvtx.annotate("psi_2", color="blue"):
                tmp_psi.append( psi[batch_num].transpose(transpose_array_default1).reshape(4,2**(N-2)) )

            transpose_array_default_batch.append(transpose_array_default2)

            end_gpu.record()
            end_gpu.synchronize()
            cumulative_time_prepare_apply_lg += cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        else:    
            # transpose the tensor with N legs to match the position of the local operator
            transpose_array_default       = np.arange(N)
            transpose_array_default[0]    = j-1
            transpose_array_default[1]    = j
            transpose_array_default[j-1]  = 0
            transpose_array_default[j]    = 1
            
            # for now psi[batch_num].operation... instead of tmp_psi[batch_num].operation
            start_gpu.record()

            with nvtx.annotate("psi_3", color="blue"):
                tmp_psi.append( psi[batch_num].transpose(transpose_array_default).reshape(4,2**(N-2)) )

            transpose_array_default_batch.append(transpose_array_default)

            end_gpu.record()
            end_gpu.synchronize()
            cumulative_time_prepare_apply_lg += cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    start_gpu.record()
    with nvtx.annotate("batched_psi", color="purple"):
        batched_psi = cp.matmul(sigmas, cp.array(tmp_psi))

    end_gpu.record()
    end_gpu.synchronize()
    cumulative_time_gemm_apply_lg += cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    #increment counter
    counter_gemm_local_gate +=1

    batched_psi = batched_psi.reshape(reshape_array_default_batch)

    tmp_psi = []

    for batch_num in range(batch_size):
        Streams[batch_num].use()
        tmp_psi.append(batched_psi[batch_num].transpose(transpose_array_default_batch[batch_num]))

    tmp_psi = cp.array(tmp_psi).reshape(batch_size,2**N)

    return tmp_psi

@nvtx.annotate("PartialTraceGeneralTensor_batch", color="red")
def PartialTraceGeneralTensor_new_GPU_batch(N, index_list, A, len_kk):
    """ Function that computes the partial trace over index_list indices (the index list needs to be ordered from smaller to bigger index)"""
    
    A_DEVICE = []
    B_DEVICE = []

    global cumulative_time_gemm_partial_trace
    global counter_gemm_partial_trace

    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    
    for kk_num in range(len_kk):

        # reshape the input vectors into tensors (here we exploit the fact that psi* is just the complex conjugate of psi )
        reshape_array_default = np.full(N,2)    
        A_initial = A.reshape(reshape_array_default)

        # generate initial transpose indices vector (we apply permutations and operatorion so transposition is correctly performed )
        list_A = np.arange(N)
        list_B = np.arange(N)

        # this changing the indeces by one is because of python stuff (the numbering starts from zero and not 1)
        index_list_tmp = np.array(index_list[kk_num]) - 1

        A_temp = np.take(list_A, index_list_tmp)
        B_temp = np.take(list_B, index_list_tmp)

        transpose2_A = np.append(A_temp, np.delete(list_A, index_list_tmp) )
        transpose2_B = np.append(np.delete(list_B, index_list_tmp), B_temp )

        ############### MAIN OPERATION AFTER ALL PREPARATION HAS BEEN PERFORMED ::: TRANSPOSITION ON A and B
        A_DEVICE.append(A_initial.transpose(transpose2_A).reshape(2**len(index_list_tmp),2**(N - len(index_list_tmp))))
        B_DEVICE.append(A_initial.transpose(transpose2_B).reshape(2**(N - len(index_list_tmp)),2**len(index_list_tmp)))

    # FINAL MULTIPLICATION
    #start_gpu.record()
    #out_DEVICE = (cp.array(A_DEVICE) @ cp.conjugate(cp.array(B_DEVICE)))
    #end_gpu.record()
    #end_gpu.synchronize()
    #cumulative_time_gemm_partial_trace += cp.cuda.get_elapsed_time(start_gpu, end_gpu)

    #increment counter
    #counter_gemm_partial_trace += 1

    #return (out_DEVICE)
    return A_DEVICE, B_DEVICE


@nvtx.annotate("Renyi2_aftergate_correct_batch", color="purple")
def Renyi2_aftergate_correct_GPU_batch(N,R,psi_DEVICE, batch_size, gate_id):
    """ compute the Renyi 2 entropy of a state psi with the subsystem being size R. This function is different from vonNeumann because it computes only 2 of the partitions that are actually relevant after the application of a local gate. Variable gate_id provides this information.  """

    # fix the R
    R = int(math.floor(R))

    # generating the indices that get traced out
    list_indic    = generate_dictionary_adjacent(N,R)
    relevant_list = []
    A_batch = []
    B_batch = []

    global Streams
    
    global cumulative_time_gemm_partial_trace
    global counter_gemm_partial_trace

    
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    
    for batch_num in range(batch_size):
        
        Streams[batch_num].use()

        gate_id_tmp = gate_id[batch_num]

        kk = []
        relevant_list_tmp = []
        
        for i in range(0,len(list_indic)):

            if (list_indic[i][0] == gate_id_tmp[1]):
                kk.append(list_indic[i])
                relevant_list_tmp.append(i)
            if (list_indic[i][R-1] == gate_id_tmp[0]):
                kk.append(list_indic[i])
                relevant_list_tmp.append(i)
        
        relevant_list.append(relevant_list_tmp)
        A, B = PartialTraceGeneralTensor_new_GPU_batch(N, kk, psi_DEVICE[batch_num], len(kk))
        A_batch.append(A)
        B_batch.append(B)

    A_batch = cp.array(A_batch)
    B_batch = cp.conjugate(cp.array(B_batch))

    start_gpu.record()
    rho = A_batch @ B_batch
    
    end_gpu.record()
    end_gpu.synchronize()
    cumulative_time_gemm_partial_trace += cp.cuda.get_elapsed_time(start_gpu, end_gpu)
    
    counter_gemm_partial_trace += 1

    ee = (-cp.log( ( (rho * rho.transpose(0,1,3,2)).sum([2,3]) ) )).real.reshape(batch_size,2)
    entropy = cp.average(ee, 1)
    
    return cp.asnumpy(entropy), cp.asnumpy(ee), relevant_list