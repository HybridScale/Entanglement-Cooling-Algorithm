import numpy as np
import cupy as cp

from pathlib import Path
from mpi4py import MPI

import pickle
import os
import dependecies as dep
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
import time
import sys


class Simulation:

    from functions import CPUIteration, GPUIteration, batchGEMMIteration

    def __init__(self, args):

        self.resume  = True if args.resume == "resume" else False

        dep.timing.set_mode(args.mode, args.timeit)

        if(self.resume):
            self.__resume_simulation(args)

        else:
            self.__new_simulation(args)


    def __new_simulation(self, args):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.batch_size = 1
        if (args.mode == "batchedGEMM"):
            self.batch_size = args.bs

        self.simulation_size = self.size * self.batch_size
        
        simulation_file = f"Renyi_entropy_raw_{args.N}_{args.R}_{args.L}_{args.MC}_{self.size}.bin"

        self.configuration = {
            "Nsites"  : args.N,
            "R"       : args.R,
            "lambda"  : args.L,
            "MCwanted": args.MC if args.MC_wanted == 0 else args.MC_wanted,
            "sim_size": self.simulation_size,
            "simfile" : simulation_file if args.o == None else args.o
        }

        #self.filename          = args.o
        self.MC                = args.MC
        self.eigen_filename_in = args.in_eigen
        self.mode              = args.mode
        self.save_eigen        = args.save_eigen

        self.states_dir  = "saved_states_{}_{}_{}".format(self.configuration["Nsites"],
                                                          self.configuration["R"], 
                                                          self.configuration["lambda"])
        if (not args.f == None):
            self.states_dir = args.f

        # model parameters
        # use periodic or open boundaries (we typically want PBC)
        self.usePBC    = True   

        # simulation parameters
        # number of eigenstates to compute (we want ground state so =1 is what we want)
        self.numval    = 1            
        self.dt        = np.pi/10.0

        # defining a logaritmically decreasing temperature grid
        self.T_grid    = np.logspace(-4,-8, num=101,base=10)

        self.__print_config(self.configuration)

        Path(self.states_dir).mkdir(exist_ok=True)
        if(self.rank == 0):
            self.__save_configuration()


    def __resume_simulation(self, args):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.configuration = self.__read_configuration( args.savedfolder)
        
        self.states_dir = args.savedfolder

        self.batch_size = 1
        if (args.mode == "batchedGEMM"):
            self.batch_size = args.bs

        self.simulation_size = self.size * self.batch_size

        self.__print_config(self.configuration)
        self.__check_mpi_size()

        self.mode      = args.mode
        self.MC        = args.MC

        # model parameters
        # use periodic or open boundaries (we typically want PBC)
        self.usePBC    = True   

        # simulation parameters
        # number of eigenstates to compute (we want ground state so =1 is what we want)
        self.numval    = 1            
        self.dt        = np.pi/10.0

        # defining a logaritmically decreasing temperature grid
        self.T_grid    = np.logspace(-4,-8, num=101,base=10)


    def start(self):
        Nsites   = self.configuration["Nsites"]
        R        = self.configuration["R"]
        MCwanted = self.configuration["MCwanted"]
        sim_rank = self.rank


        if (self.resume):
            state, ee, y, cc, MCold= self.__resume_read_states(sim_rank)
            #print(f"y shape {y.shape}")
            #MCold = y.shape[1] + 1

            MCwanted = self.configuration["MCwanted"]
            self.MC = MCwanted if self.MC == None else self.MC + MCold

        else:
            state = self.__prepareEigenvectors(sim_rank)

            ent, ee = dep.Renyi2(Nsites, R, state)
    
            y = np.empty(0)
            MCold = 1
            cc = 0

        list_pairs_local_ham = dep.generate_dictionary(Nsites, 1, True)

        dep.timing.Start("TotalTime")
        y = self.__MC_Simulation(sim_rank, cc, MCold, list_pairs_local_ham, state, y, ee)
        dep.timing.Stop("TotalTime")

        self.comm.Barrier()

        if(self.rank == 0):
            print(dep.timing)

    def __check_mpi_size(self):
        simulation_size = self.simulation_size
        MPIresumeSize   = self.configuration["sim_size"]

        if ( MPIresumeSize != simulation_size):
            if (self.rank == 0):
                print(f"invalid number of MPI ranks!! Simulation started with {simulation_size} ranks instead of {MPIresumeSize}.")
            sys.exit()

        
    def __print_config(self, config):
        if(self.rank == 0):
            max_length = 0
            for key in config:
                current_length = len( str(config[key])) + 12 
                max_length = current_length if current_length > max_length else max_length
            print(f'{"Configuration:":-^{max_length }}')
            for key in config:
                print(f'|{key:<10}{config[key]:>{max_length - 12}}|')

            print(f'{"":-^{max_length}}')
        
        self.comm.Barrier()


    def __read_configuration(self, folder):
        with open(folder + "/configuration", "rb") as f:
            configuration = pickle.load(f)
        
        return configuration

        
    def __save_configuration(self):
        with open(self.states_dir + "/configuration", "wb") as f:
            pickle.dump(self.configuration, f)


    def __prepareEigenvectors(self, sim_rank):

        Nsites = self.configuration["Nsites"]
        R      = self.configuration["R"]
        Lambda = self.configuration["lambda"]

        def doApplyHamClosed(psiIn):
            """ supplementary function  cast the Hamiltonian 'H' as a linear operator """
            return dep.doApplyHamTENSOR(psiIn, hloc, Nsites, self.usePBC)

        if (not self.resume and self.eigen_filename_in):
            with open(self.eigen_filename_in, "rb") as f:
                eigenvectors = pickle.load(f)

        else:
            if(self.rank == 0):
                print("Calculating initial eigenvectors")
                hloc = dep.tfim_LocalHamiltonian_new(Lambda)
                H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)
                eigenvalues, eigenvectors = eigsh(H, k=self.numval, which='SA',return_eigenvectors=True)
                eigenvectors = eigenvectors[:, 0].astype("complex128")
                
                if(self.save_eigen):
                    with open(f"eigenvecs_{Nsites}_{R}.bin", "wb") as file:
                        pickle.dump(eigenvectors, file)
                        print("Eigenstate saved in file", file.name)

            else:
                eigenvectors = np.empty(2**Nsites, dtype="complex128")
                
            self.comm.Barrier()

            eigenvectors = self.comm.bcast(eigenvectors, root=0)
        
        return np.array(eigenvectors)


    def __resume_read_states(self, sim_rank):


        if (self.mode == "batchedGEMM"):
            Nsites   = self.configuration["Nsites"]

            state = np.empty(0, dtype = "complex128")
            ee    = np.empty(0)
            y     = np.empty(0)

            for batch_num in range(self.batch_size):
                with open(self.states_dir + f"/state_{sim_rank * self.batch_size + batch_num}", "rb") as f:
                    state = np.append(state, pickle.load(f))
                    ee    = np.append(ee, pickle.load(f))
                    ytmp  = pickle.load(f)
                    steps = ytmp.shape[0]
                    y     = np.append(y, ytmp)
                    cc    = pickle.load( f)

            state = state.reshape(self.batch_size, 2**Nsites)
            ee    = ee.reshape(self.batch_size, Nsites)
            y     = y.reshape(self.batch_size, steps)

        else:
            with open(self.states_dir + f"/state_{sim_rank}", "rb") as f:
                state   = pickle.load(f)
                ee      = pickle.load( f)
                y       = pickle.load( f)
                cc      = pickle.load( f)
                steps   = y.shape[0]

        return state, ee, y, cc, (steps +1)


    def __cudaDeviceID(self):
            node_comm = self.comm.Split_type(MPI.COMM_TYPE_SHARED)
            node_rank = node_comm.Get_rank()
            node_size = node_comm.Get_size()
            number_of_device_on_node = cp.cuda.runtime.getDeviceCount()

            # set device id to node ranks
            device_id  = node_rank % number_of_device_on_node
            return device_id


    def __MC_Simulation(self, simID, cc, MCold, loc_pairs, state, y, ee):

        MCsteps = self.MC
    
        MCsteps_exponent = round(np.log10(MCsteps-MCold))
        if (MCsteps_exponent > 4):
            print_exponent = int( 1000 )
        elif (MCsteps_exponent <= 0):
            print_exponent = 1  
        else:
            print_exponent = int( 10**(MCsteps_exponent-1) )  #Printf only 10 times which steps computing

        if (self.mode == "CPU"):
            y, accepted = self.CPUIteration(simID, cc, MCold, print_exponent, loc_pairs, state, y, ee )
        elif (self.mode == "GPU"):
            deviceID = self.__cudaDeviceID()
            with cp.cuda.Device(deviceID):
                y, accepted = self.GPUIteration(simID, cc, MCold, print_exponent, loc_pairs, state, y, ee)
        elif (self.mode == "batchedGEMM"):
            deviceID = self.__cudaDeviceID()
            with cp.cuda.Device(deviceID):
                if (not self.resume):
                    state  = cp.tile(state, (self.batch_size, 1))
                    ee     = cp.tile(ee, (self.batch_size, 1))
                y = self.batchGEMMIteration(simID, cc, MCold, print_exponent, loc_pairs, state, y, ee)


        return np.array(y)