import numpy as np
import cupy as cp
from mpi4py import MPI
import pickle
import os
from dep import *
from scipy.sparse.linalg import LinearOperator, eigsh, eigs

lista_y = np.empty(0)

def MC_Simulation(x, MCsteps, MCsteps_old, loc_paris , state, ee_list_old, Nsites, dt, R, T_grid):
    accepted = 0
    cc = 0
    global lista_y
    
    MCsteps_exponent = round(np.log10(MCsteps-MCsteps_old))
    
    if (MCsteps_exponent >4):
        print_exponent = int( 1000 )  
    else:
        print_exponent = int( 10**(MCsteps_exponent-1) )  #Printf only 10 times which steps computing

    cum_time_apply_local_gate=0
    cum_time_renyi=0
    
    for yy in range(MCsteps_old, MCsteps+1):
        random_pair, random_pair_id = select_cooling_evolution_indices(loc_paris)
        random_gate, random_gate_id = select_cooling_evolution_gates()
        start=time.time()
        new_state = ApplyLocalGate(random_gate,random_pair,state,Nsites,2,dt)
        end=time.time()
        cum_time_apply_local_gate+= (end-start)

        start=time.time()
        ent_new, ee_list_new, relevant_partitions = Renyi2_aftergate_correct(Nsites,R,new_state,random_pair)
        end=time.time()
        cum_time_renyi+= (end-start)

        ent_tamp = ee_list_old.copy()

        ent_tamp[relevant_partitions[0]] = ee_list_new[0]
        ent_tamp[relevant_partitions[1]] = ee_list_new[1]

        old_value = np.average(ee_list_old)
        new_value = np.average(ent_tamp)


        # we avoid overflow
        p_exp = -(1.0/T_grid[cc])*(new_value - old_value)
        if (p_exp > 50):
            p = 1.0
        else:
            p = np.exp(p_exp)

        random_value = np.random.uniform(0,1)
        if (random_value <= min(1.0,p)):
            accepted += 1
            state = new_state.copy()
            ee_list_old = ent_tamp.copy()


        #lista_y.append(np.average(ee_list_old))
        lista_y = np.append(lista_y, np.average(ee_list_old))

        if( yy %  print_exponent == 0 ):
            print("rank", x, 'MC step = %d' % yy)

        if( yy % 1000 == 0 and MCsteps>MCsteps_old):
            save_state_file_name = "state_{}".format(x)
            f = open(str(states_dir)+"/"+str(save_state_file_name), "wb")
            pickle.dump(new_state, f)
            pickle.dump(ee_list_old, f)
            pickle.dump(lista_y, f)
            f.close()
            print("rank", x, "saved state on step", yy)

        if (yy % int(MCsteps/100) == 0):
    	    cc += 1
        
        #if ( yy % 1000 ==0 ):
        #    print("rank",x,"cum_time_apply_local_gate: " + str(round(cum_time_apply_local_gate,4)), file=open("time_of_exe/sim_{}.out".format(x),"a"))
        #    print("rank",x,"cum_time_partial_trace in Renyi2 after gate: " + str(round(config.cum_time_partial_trace,4)), file=open("time_of_exe/sim_{}.out".format(x),"a"))
        #    print("rank",x,"cum_time_entropy in Renyi2 after gate: " + str(round(config.cum_time_entropy,4) ), file=open("time_of_exe/sim_{}.out".format(x),"a"))
        #    print("rank",x,"cum_time_Renyi2: " + str(round(cum_time_renyi,4)), file=open("time_of_exe/sim_{}.out".format(x),"a"))
        #    print(str(round(cum_time_apply_local_gate,4)) +" + "+ str(round(cum_time_renyi,4)) +" = "+str(round(cum_time_apply_local_gate+ cum_time_renyi,4) ) +"\n\n" , file=open("time_of_exe/sim_{}.out".format(x),"a"))
        #    print(yy, round(cum_time_apply_local_gate+ cum_time_renyi,4),file=open("time_of_exe/sim_plot_{}.out".format(x),"a"))
        
def MC_Simulation_GPU(x, node_rank, MCsteps, MCsteps_old, loc_pairs , state, ee_list_old, Nsites, dt, R, T_grid):
    accepted = 0
    cc = 0
    global lista_y
    
    MCsteps_exponent = round(np.log10(MCsteps-MCsteps_old))
    
    if (MCsteps_exponent >4):
        print_exponent = int( 1000 )  
    else:
        print_exponent = int( 10**(MCsteps_exponent-1) )  #Printf only 10 times which steps computing

    cum_time_apply_local_gate=0
    cum_time_renyi=0
    
    state_DEVICE = cp.asarray(state)
    sigma_list_device = create_sigma_list_GPU(dt)
    rand_sigma_value = len(sigma_list_device)
    
    for yy in range(MCsteps_old, MCsteps+1):
        random_pair, random_pair_id = select_cooling_evolution_indices(loc_pairs)
        #random_gate, random_gate_id = select_cooling_evolution_gates()

        random_sigma_gate = sigma_list_device[np.random.randint(rand_sigma_value)]

        new_state_DEVICE = ApplyLocalGate_GPU(random_sigma_gate,random_pair,state_DEVICE,Nsites,2,dt)
        
        ent_new_DEVICE, ee_list_new_DEVICE, relevant_partitions = Renyi2_aftergate_correct_GPU(Nsites,R, new_state_DEVICE,random_pair)
        
        ent_new = cp.asnumpy(ent_new_DEVICE)
        ee_list_new = cp.asnumpy(ee_list_new_DEVICE)

        ent_tamp = ee_list_old.copy()

        ent_tamp[relevant_partitions[0]] = ee_list_new[0]
        ent_tamp[relevant_partitions[1]] = ee_list_new[1]

        old_value = np.average(ee_list_old)
        new_value = np.average(ent_tamp)

        # we avoid overflow
        p_exp = -(1.0/T_grid[cc])*(new_value - old_value)
        if (p_exp > 50):
            p = 1.0
        else:
            p = np.exp(p_exp)

        random_value = np.random.uniform(0,1)
        if (random_value <= min(1.0,p)):
            accepted += 1
            state_DEVICE = new_state_DEVICE.copy()
            ee_list_old = ent_tamp.copy()


        #lista_y.append(np.average(ee_list_old))
        lista_y = np.append(lista_y, np.average(ee_list_old))

        if( yy %  print_exponent == 0 ):
            print("simulation", x, 'MC step = %d' % yy, "on", socket.gethostname())

        if( yy % 100000 == 0 and MCsteps>MCsteps_old):
            new_state = cp.asnumpy(new_state_DEVICE)
            save_state_file_name = "state_{}".format(x)
            f = open(str(states_dir)+"/"+str(save_state_file_name), "wb")
            pickle.dump(new_state, f)
            pickle.dump(ee_list_old, f)
            pickle.dump(lista_y, f)
            f.close()
            print("rank", x, "saved state on step", yy)

        if (yy % int(MCsteps/100) == 0):
    	    cc += 1

def cpu_sim(args):
    # this dumb function doesn't work when in dependecies.py file so we put it here
    def doApplyHamClosed(psiIn):
        """ supplementary function  cast the Hamiltonian 'H' as a linear operator """
        return doApplyHamTENSOR(psiIn, hloc, Nsites, usePBC)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Nsites = args.N
    R = args.R
    lambdaa = args.L
    MCsteps = args.MC
    filename = args.o
    eigen_filename_in = args.in_eigen

    # model parameters
    usePBC    = True                         # use periodic or open boundaries (we typically want PBC)

    # simulation parameters
    numval    = 1                            # number of eigenstates to compute (we want ground state so =1 is what we want)
    dt        = np.pi/10.0                   # 'time-evolution' time step (this is a parameter we should change a bit to see if the results are consistent with it)

    # defining a logaritmically decreasing temperature grid
    T_grid    = np.logspace(-4,-8, num=101,base=10)
    #print('T_grid =', T_grid)

    sX, sY, sZ, sI = PauliMatrice()

    concurrence1 = []
    concurrence2 = []
    magz_exact = []
    magz_num = []
    global lista_y

    global states_dir

    states_dir  = "saved_states_{}_{}_{}".format(Nsites, R, lambdaa)
     
    if ( args.resume):
        if(not os.path.exists(states_dir) ):
            if(rank == 0):
                print("Error: no saved states, try without flag --resume.", flush=True)
                comm.Abort()
        comm.Barrier()
        
        number_of_saved_states = len(os.listdir(states_dir))

        if( number_of_saved_states != size ):
            if( rank ==0 ):
                print("Error: number of parallel simulations is different than number of saved states. Try with mpirun -n", number_of_saved_states)
                print("or set the arguments to calculate the simulations from the beginning (without --resume)", flush=True)
                comm.Abort()
        comm.Barrier()

        saved_state_file = open(str(states_dir)+"/state_"+str(rank), "rb")
        
        eigenvectors = pickle.load( saved_state_file)
        ee_list = pickle.load( saved_state_file)
        lista_y = np.append(lista_y, pickle.load( saved_state_file))
     
        saved_state_file.close()

        MCsteps_old = len(lista_y)
        MCsteps_to_cont = MCsteps - MCsteps_old

        if( MCsteps_to_cont < 10):
            if( rank ==0 ):
                print("Insert bigger number of monte carlo steps (MC) to resume simulations >", MCsteps_old+10)
            comm.Barrier()
            print("Current MC state of simulation ", rank, " is: ", MCsteps_old)
            comm.Abort()
        comm.Barrier()

        print("rank ", rank, " loaded : " , lista_y.shape[0], ", number of steps to continue: ", MCsteps_to_cont, flush=True)

        list_pairs_local_ham = generate_dictionary(Nsites,1, True)

        MC_Simulation(rank, MCsteps, MCsteps_old, list_pairs_local_ham, eigenvectors, ee_list, Nsites, dt, R, T_grid)    
     
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(np.array(lista_y), MC_data_final, op=MPI.SUM, root=0)
    
        if (rank == 0):
            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()

    if (not args.resume):

        if (args.in_eigen):
            ###Read input eigenvector
            f = open(eigen_filename_in, "rb")
            eigenvectors = pickle.load(f)
            f.close()
        
        else:
            if(rank == 0):
                #rank 0 calculate inital eigenvectors (save if flag --save_eigen) and distribute to other ranks
                print("Calculating initial eigenvectors")
                hloc = tfim_LocalHamiltonian_new(lambdaa)
                H = LinearOperator((2**Nsites, 2**Nsites), matvec=doApplyHamClosed)
                eigenvalues, eigenvectors = eigsh(H, k=numval, which='SA',return_eigenvectors=True)
                
                if(args.save_eigen):
                    save_eigen_file = open("eigenvecs_{}_{}.bin".format(Nsites, R), "wb")
                    pickle.dump(eigenvectors, save_eigen_file)
                    save_eigen_file.close()
                    print("Eigenstate saved in file", save_eigen_file.name)

            else:
                eigenvectors = np.empty(2**Nsites, dtype="float64")    
            
            comm.Barrier()

        eigenvectors = comm.bcast(eigenvectors, root=0)

        if (rank == 0):
                if not os.path.exists(states_dir):
                    os.mkdir(states_dir)    


        #computing the initial Renyi2 entropy
        ent, ee_list = Renyi2(Nsites,R,eigenvectors[:,0])
    
        concurrence1.append(ent)
    
        # generate the list of pairs for the application of the local Hamiltonian
        list_pairs_local_ham = generate_dictionary(Nsites,1, True)
    
        print("Starting simulation", flush=True)
        MC_Simulation(rank, MCsteps, 1, list_pairs_local_ham, eigenvectors[:,0], ee_list, Nsites, dt, R, T_grid)
    
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(lista_y, MC_data_final, op=MPI.SUM, root=0)
    
        if (rank == 0):
            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()

def gpu_sim(args):
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    Nsites = args.N
    R = args.R
    lambdaa = args.L
    MCsteps = args.MC
    filename = args.o
    eigen_filename_in = args.in_eigen

    #if ( rank == 0):
    #    print(sys.argv[0])

    # model parameters
    usePBC    = True                         # use periodic or open boundaries (we typically want PBC)

    # simulation parameters
    numval    = 1                            # number of eigenstates to compute (we want ground state so =1 is what we want)
    dt        = np.pi/10.0                   # 'time-evolution' time step (this is a parameter we should change a bit to see if the results are consistent with it)

    # defining a logaritmically decreasing temperature grid
    T_grid    = np.logspace(-4,-8, num=101,base=10)
    #print('T_grid =', T_grid)

    sX, sY, sZ, sI = PauliMatrice()

    concurrence1 = []
    concurrence2 = []
    magz_exact = []
    magz_num = []
    global lista_y

    global states_dir

    states_dir  = "saved_states_{}_{}_{}".format(Nsites, R, lambdaa)
     
    if ( args.resume):
        if(not os.path.exists(states_dir) ):
            if(rank == 0):
                print("Error: no saved states, try without flag --resume.", flush=True)
                comm.Abort()
        comm.Barrier()
        
        number_of_saved_states = len(os.listdir(states_dir))

        if( number_of_saved_states != size ):
            if( rank ==0 ):
                print("Error: number of parallel simulations is different than number of saved states. Try with mpirun -n", number_of_saved_states)
                print("or set the arguments to calculate the simulations from the beginning (without --resume)", flush=True)
                comm.Abort()
        comm.Barrier()

        saved_state_file = open(str(states_dir)+"/state_"+str(rank), "rb")
        
        eigenvectors = pickle.load( saved_state_file)
        ee_list = pickle.load( saved_state_file)
        lista_y = np.append(lista_y, pickle.load( saved_state_file))
        
        saved_state_file.close()

        MCsteps_old = len(lista_y)
        MCsteps_to_cont = MCsteps - MCsteps_old

        if( MCsteps_to_cont < 10):
            if( rank ==0 ):
                print("Insert bigger number of monte carlo steps (MC) to resume simulations >", MCsteps_old+10)
            comm.Barrier()
            print("Current MC state of simulation ", rank, " is: ", MCsteps_old)
        
            comm.Abort()
        comm.Barrier()

        print("rank ", rank, " loaded : " , lista_y.shape[0], ", number of steps to continue: ", MCsteps_to_cont, flush=True)

        list_pairs_local_ham = generate_dictionary(Nsites,1, True)

        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        node_rank = node_comm.Get_rank()
        node_size = node_comm.Get_size()
        number_of_device_on_node = cp.cuda.runtime.getDeviceCount()
        cp.cuda.Device(node_rank % number_of_device_on_node).use()

        MC_Simulation_GPU(rank, node_rank, MCsteps, MCsteps_old, list_pairs_local_ham, eigenvectors, ee_list, Nsites, dt, R, T_grid)    
     
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(np.array(lista_y), MC_data_final, op=MPI.SUM, root=0)
    
        if (rank == 0):
            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()

    if (not args.resume):

        if (args.in_eigen):
            ###Read input eigenvector
            f = open(eigen_filename_in, "rb")
            eigenvectors = pickle.load(f)
            f.close()

        if (rank == 0):
                if not os.path.exists(states_dir):
                    os.mkdir(states_dir)    


        #computing the initial Renyi2 entropy
        
        ent, ee_list = Renyi2(Nsites,R,eigenvectors[:,0])
    
        concurrence1.append(ent)
    
        # generate the list of pairs for the application of the local Hamiltonian
        list_pairs_local_ham = generate_dictionary(Nsites,1, True)

        node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        node_rank = node_comm.Get_rank()
        node_size = node_comm.Get_size()
        number_of_device_on_node = cp.cuda.runtime.getDeviceCount()
        print(rank, node_rank, node_rank % number_of_device_on_node)

        on_device  = node_rank % number_of_device_on_node
        with cp.cuda.Device(on_device):
            print("node rank", node_rank, "Starting simulation on device", on_device,flush=True)
            MC_Simulation_GPU(rank, node_rank, MCsteps, 1, list_pairs_local_ham, eigenvectors[:,0], ee_list, Nsites, dt, R, T_grid)
    
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(lista_y, MC_data_final, op=MPI.SUM, root=0)
    
        if (rank == 0):
            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()
            
def batch_sim_gpu(args):
    print(args)
    #fun_fun(args.sim_per_batch)
	    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
	
    if(size != args.sim_per_batch):
        print("Number of simulation different than -s option", flush=True)
        comm.Abort()
    comm.Barrier()
		

    Nsites = args.N
    R = args.R
    lambdaa = args.L
    MCsteps = args.MC
    filename = args.o
    eigen_filename_in = args.in_eigen

    # model parameters
    usePBC    = True                         # use periodic or open boundaries (we typically want PBC)

    # simulation parameters
    numval    = 1                            # number of eigenstates to compute (we want ground state so =1 is what we want)
    dt        = np.pi/10.0                   # 'time-evolution' time step (this is a parameter we should change a bit to see if the results are consistent with it)

    # defining a logaritmically decreasing temperature grid
    T_grid    = np.logspace(-4,-8, num=101,base=10)
    #print('T_grid =', T_grid)

    sX, sY, sZ, sI = PauliMatrice()

    concurrence1 = []
    concurrence2 = []
    magz_exact = []
    magz_num = []
    global lista_y

    global states_dir

    states_dir  = "saved_states_{}_{}_{}".format(Nsites, R, lambdaa)

    if (not args.resume):

        if (args.in_eigen):
            ###Read input eigenvector
            f = open(eigen_filename_in, "rb")
            eigenvectors = pickle.load(f)
            f.close()

        if (rank == 0):
                if not os.path.exists(states_dir):
                    os.mkdir(states_dir)    


        #computing the initial Renyi2 entropy
        
        ent, ee_list = Renyi2(Nsites,R,eigenvectors[:,0])
    
        concurrence1.append(ent)
    
        # generate the list of pairs for the application of the local Hamiltonian
        list_pairs_local_ham = generate_dictionary(Nsites,1, True)

        #node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
        #node_rank = node_comm.Get_rank()
        #node_size = node_comm.Get_size()
        #number_of_device_on_node = cp.cuda.runtime.getDeviceCount()
        #print(rank, node_rank, node_rank % number_of_device_on_node)

        batch_rank = rank + args.sim_per_batch * args.batch_job_id
        print("rank", rank, "batch rank", batch_rank, "number of devices", cp.cuda.runtime.getDeviceCount())
        print("node rank", batch_rank, "Starting simulation on device", flush=True)
        MC_Simulation_GPU(batch_rank, args.batch_job_id, MCsteps, 1, list_pairs_local_ham, eigenvectors[:,0], ee_list, Nsites, dt, R, T_grid)
    
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(lista_y, MC_data_final, op=MPI.SUM, root=0)
    
        if (rank == 0):
            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()
