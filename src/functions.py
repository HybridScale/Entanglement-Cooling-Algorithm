import numpy as np
import cupy as cp
from mpi4py import MPI
import pickle
import os
import dependecies as dep
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
import time
import sys

cumulative_time_apply_local_gate = 0
cumulative_time_renyi            = 0


def MC_Simulation(x, MCsteps, MCsteps_old, loc_paris , state, ee_list_old, Nsites, dt, R, T_grid, lista_y):
    accepted = 0
    cc = 0
    
    global cumulative_time_apply_local_gate
    global cumulative_time_renyi

    dep.counter_gemm_partial_trace = 0

    
    MCsteps_exponent = round(np.log10(MCsteps-MCsteps_old))
    
    if (MCsteps_exponent > 4):
        print_exponent = int( 1000 )
    elif (MCsteps_exponent <= 0):
        print_exponent = 1  
    else:
        print_exponent = int( 10**(MCsteps_exponent-1) )  #Printf only 10 times which steps computing
    
    for yy in range(MCsteps_old, MCsteps+1):
        random_pair, random_pair_id = dep.select_cooling_evolution_indices(loc_paris)
        random_gate, random_gate_id = dep.select_cooling_evolution_gates()

        start= MPI.Wtime()
        new_state = dep.ApplyLocalGate(random_gate,random_pair,state,Nsites,2,dt)
        cumulative_time_apply_local_gate+= MPI.Wtime()-start


        start=MPI.Wtime()
        ent_new, ee_list_new, relevant_partitions = dep.Renyi2_aftergate_correct(Nsites,R,new_state,random_pair)
        cumulative_time_renyi+= MPI.Wtime()-start

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

        if ( MCsteps >=100 and yy % int(MCsteps/100) == 0):
    	    cc += 1

    return np.array(lista_y)

        
def MC_Simulation_GPU(x, MCsteps, MCsteps_old, loc_pairs , state, ee_list_old, Nsites, dt, R, T_grid, lista_y):
    accepted = 0
    cc = 0

    global cumulative_time_apply_local_gate
    global cumulative_time_renyi

    dep.counter_gemm_partial_trace = 0

    MCsteps_exponent = round(np.log10(MCsteps-MCsteps_old))
    
    if (MCsteps_exponent >4):
        print_exponent = int( 1000 )  
    else:
        print_exponent = int( 10**(MCsteps_exponent-1) )  #Printf only 10 times which steps computing

    
    state_DEVICE = cp.asarray(state)

    # Create and return sigma gates on device (cupy array)
    sigma_list_device = dep.create_sigma_list_GPU(dt)
    
    rand_sigma_value = len(sigma_list_device)
    
    for yy in range(MCsteps_old, MCsteps+1):
        random_pair, random_pair_id = dep.select_cooling_evolution_indices(loc_pairs)
        random_sigma_gate = sigma_list_device[np.random.randint(rand_sigma_value)]

        
        start_gpu.record()
        new_state_DEVICE = dep.ApplyLocalGate_GPU(random_sigma_gate,random_pair,state_DEVICE,Nsites,2,dt)
        end_gpu.record()
        end_gpu.synchronize()
        cumulative_time_apply_local_gate += cp.cuda.get_elapsed_time(start_gpu, end_gpu)

        start_gpu.record()      
        ent_new_DEVICE, ee_list_new_DEVICE, relevant_partitions = dep.Renyi2_aftergate_correct_GPU(Nsites,R, new_state_DEVICE,random_pair)
        end_gpu.record()
        end_gpu.synchronize()
        cumulative_time_renyi += cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        
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

        lista_y = np.append(lista_y, np.average(ee_list_old))

        if( yy %  print_exponent == 0 ):
            print("simulation", x, 'MC step = %d' % yy)

        if( yy % 1000 == 0 and MCsteps>MCsteps_old):
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

    # calculate all timing in s as they are returned in ms
    cumulative_time_apply_local_gate /= 1000
    cumulative_time_renyi /= 1000
    dep.cumulative_time_gemm_apply_lg /= 1000
    dep.cumulative_time_prepare_apply_lg /= 1000
    dep.cumulative_time_gemm_partial_trace /= 1000
            
    return cp.asnumpy(cp.array(lista_y))


def MC_Simulation_GPU_batch(x, batch_size, MCsteps, MCsteps_old, loc_pairs , state, ee_list_old, Nsites, dt, R, T_grid, lista_y):
    accepted = 0
    cc = 0

    global cumulative_time_apply_local_gate
    global cumulative_time_renyi

    dep.counter_gemm_partial_trace = 0

    MCsteps_exponent = round(np.log10(MCsteps-MCsteps_old))
    
    if (MCsteps_exponent >4):
        print_exponent = int( 1000 )  
    else:
        print_exponent = int( 10**(MCsteps_exponent-1) )  #Printf only 10 times which steps computing

    
    states = []
    ee_list_old_list = []
    if (len(state.shape) == 1):
        for _ in range(batch_size):
            states.append(state)
            ee_list_old_list.append( ee_list_old)
    else:
        for batch_num in range(batch_size):
            states.append(state[batch_num])
            ee_list_old_list.append( ee_list_old[batch_num])
        
        
    states = cp.asarray( states, dtype="complex_")
    ee_list_old_list = np.asarray( ee_list_old_list)
    
    sigma_list_device = dep.create_sigma_list_GPU(dt)
    
    for yy in range(MCsteps_old, MCsteps+1):
        random_pairs, random_pair_ids = dep.select_cooling_evolution_indices_batch(loc_pairs, batch_size)

        random_sigma_gates = dep.select_sigma_gates_batch(sigma_list_device, batch_size)

        start_gpu.record()      
        new_states = dep.ApplyLocalGate_GPU_batch(random_sigma_gates, 
                                                  batch_size ,
                                                  random_pairs, 
                                                  states, 
                                                  Nsites, 
                                                  2, 
                                                  dt)
        end_gpu.record()
        end_gpu.synchronize()
        cumulative_time_apply_local_gate+= cp.cuda.get_elapsed_time(start_gpu, end_gpu)  

        start_gpu.record()      
        ents_new, ees_list_new, relevant_partitions = dep.Renyi2_aftergate_correct_GPU_batch(Nsites, 
                                                                                             R, 
                                                                                             new_states, 
                                                                                             batch_size,
                                                                                             random_pairs)
        end_gpu.record()
        end_gpu.synchronize()
        cumulative_time_renyi += cp.cuda.get_elapsed_time(start_gpu, end_gpu)  
        
        ent_tamp = ee_list_old_list.copy()
        for batch_num in range(batch_size):             

            ent_tamp[batch_num] [relevant_partitions[batch_num][0]] = ees_list_new[batch_num][0]
            ent_tamp[batch_num] [relevant_partitions[batch_num][1]] = ees_list_new[batch_num][1]

            old_value = np.average( ee_list_old_list[batch_num])
            new_value = np.average( ent_tamp[batch_num])
            
            p_exp = -(1.0/T_grid[cc])*(new_value - old_value)
            if (p_exp > 50):
                p = 1.0
            else:
                p = np.exp(p_exp)

            random_value = np.random.uniform(0,1)
            if (random_value <= min(1.0,p)):
                accepted += 1
                states[batch_num] = new_states[batch_num].copy()
                ee_list_old_list[batch_num] = ent_tamp[batch_num].copy()

            if(yy == 1):
                lista_y.append( np.average(ee_list_old_list[batch_num]))
            else:
                lista_y[batch_num] = np.append(lista_y[batch_num], np.average(ee_list_old_list[batch_num]))
                #print(np.average(ee_list_old_list[0]))
            
        if( yy %  print_exponent == 0 ):
            print("rank", x, 'MC step = %d' % yy)

        if( yy % 1000 == 0 and MCsteps>MCsteps_old):
            for batch_num in range(batch_size):
                save_state_file_name = "state_{}".format(x*batch_size + batch_num)
                f = open(str(states_dir)+"/"+str(save_state_file_name), "wb")
                pickle.dump(states[batch_num].get(), f)
                pickle.dump(ee_list_old_list[batch_num], f)
                pickle.dump(lista_y[batch_num], f)
                f.close()
            print("rank", x, "saved state on step", yy)

        if (yy % int(MCsteps/100) == 0):
            cc += 1
    
    # calculate all timing in s as they are returned in ms
    cumulative_time_apply_local_gate /= 1000
    cumulative_time_renyi /= 1000
    dep.cumulative_time_gemm_apply_lg /= 1000
    dep.cumulative_time_gemm_partial_trace /= 1000
    
    #print(cp.asnumpy(lista_y))
            
    return cp.asnumpy(lista_y)

def print_times(time_lg_max, time_renyi_max, total_time):
    print("######")
    print("time: ", )
    print('{:30s} {:5.4f}'.format("apply local gate:",time_lg_max ))
    print('{:30s} {:5.4f}'.format("gemm apply local gate:",dep.cumulative_time_gemm_apply_lg ))
    print('{:30s} {:5.4f}'.format("prepare apply local gate:",dep.cumulative_time_prepare_apply_lg ))
    print('{:30s} {:5.0f}'.format("counter gemm apply lg:",dep.counter_gemm_local_gate ))
    print('\n{:30s} {:5.4f}'.format("renyi2: ", time_renyi_max ))
    print('{:30s} {:5.4f}'.format("gemm partial trace:",dep.cumulative_time_gemm_partial_trace ))
    print('{:30s} {:5.0f}'.format("counter gemm partial trace:",dep.counter_gemm_partial_trace ))
    print('\n{:30s} {:5.4f}'.format("total time:",total_time ))

def print_times_csv_format(arguments, procedure_size, time_lg_max, time_renyi_max, total_time):
    print(arguments.mode,
          procedure_size,
          arguments.N,
          arguments.R,
          arguments.L,
          arguments.MC,
          time_lg_max,
          dep.cumulative_time_gemm_apply_lg,
          dep.cumulative_time_prepare_apply_lg,
          dep.counter_gemm_local_gate,
          time_renyi_max,
          dep.cumulative_time_gemm_partial_trace,
          dep.counter_gemm_partial_trace,
          total_time , sep =", ")

def create_and_append_output():
    # Create file output.csv if does not already exists
    if (not os.path.exists("output.csv")):
        f = open("output.csv", "a")
        # Print in files header/column names
        print("Mode, ",
              "Procedure_size, ",
              "N, ",
              "R, ",
              "L, ",
              "MC, ",
              "Apply_local_gate, ",
              "Apply_local_gate_gemm, ",
              "Apply_local_gate_prepare, ",
              "Appyl_local_gate_#gemm, ",
              "Renyi2, ",
              "Partial_trace_gemm, ",
              "Partial_trace_#gemm, ",
              "Total_time, ",
               file=f)
        f.close()


def set_simulations(args):      
    def doApplyHamClosed(psiIn):
        """ supplementary function  cast the Hamiltonian 'H' as a linear operator """
        return dep.doApplyHamTENSOR(psiIn, hloc, Nsites, usePBC)
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    Nsites = args.N
    R = args.R
    lambdaa = args.L
    MCsteps = args.MC
  
    filename = args.o
    eigen_filename_in = args.in_eigen
    
    if (args.mode == "batchedGEMM"):
        batch_size = args.bs

    # model parameters
    usePBC    = True                         # use periodic or open boundaries (we typically want PBC)

    # simulation parameters
    numval    = 1                            # number of eigenstates to compute (we want ground state so =1 is what we want)
    dt        = np.pi/10.0                   # 'time-evolution' time step (this is a parameter we should change a bit to see 
                                             # if the results are consistent with it)

    if( args.mode != "CPU"):
        cp.matmul(cp.random.random(1_000), cp.random.random(1_000))

    # defining a logaritmically decreasing temperature grid
    T_grid    = np.logspace(-4,-8, num=101,base=10)
    #print('T_grid =', T_grid)

    sX, sY, sZ, sI = dep.PauliMatrice()

    concurrence1 = []
    concurrence2 = []
    magz_exact = []
    magz_num = []

    global states_dir

    global cumulative_time_apply_local_gate
    global cumulative_time_renyi

    global start_gpu
    global end_gpu

    states_dir  = "saved_states_{}_{}_{}".format(Nsites, R, lambdaa)
    lista_y     = []
    
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
                hloc = dep.tfim_LocalHamiltonian_new(lambdaa)
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
        
        
        # check for/create saved states folder 
        if (rank == 0):
            create_and_append_output()

            if not os.path.exists(states_dir):
                os.mkdir(states_dir)  
    
        start = time.time()
        
        #computing the initial Renyi2 entropy
        ent, ee_list = dep.Renyi2(Nsites,R,eigenvectors[:,0])
    
        concurrence1.append(ent)
    
        # generate the list of pairs for the application of the local Hamiltonian
        list_pairs_local_ham = dep.generate_dictionary(Nsites,1, True)
        
        array_job_rank = rank + size * args.array_job_id
        
        start = time.time()

        if( (args.mode == "GPU") or (args.mode == "batchedGEMM")):
            node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            node_rank = node_comm.Get_rank()
            node_size = node_comm.Get_size()
            number_of_device_on_node = cp.cuda.runtime.getDeviceCount()

            # set device id to node ranks
            device_id  = node_rank % number_of_device_on_node 

            # setting device to use
            with cp.cuda.Device(device_id):
                # Create global objects cuda event for measurment of time exe
                # Created after setting working device per MPI process 
                # lazy implementation 

                start_gpu = cp.cuda.Event()
                end_gpu = cp.cuda.Event()
                print("node rank", node_rank, "Starting simulation on device", cp.cuda.runtime.getDeviceProperties(device_id)['name'] ,flush=True)
               
                if(args.mode == "GPU"):
                    lista_y = MC_Simulation_GPU(array_job_rank, MCsteps, 1, list_pairs_local_ham, eigenvectors[:,0], ee_list, Nsites, dt, R, T_grid, lista_y)           
               
                elif(args.mode == "batchedGEMM"):
                    dep.set_streams_global(batch_size)
                    lista_y = MC_Simulation_GPU_batch(array_job_rank, batch_size, MCsteps, 1, list_pairs_local_ham, eigenvectors[:,0], ee_list, Nsites, dt, R, T_grid, lista_y)
           
                    #average by number of batch size
                    lista_y = np.average(lista_y, axis=0)

        if ( args.mode == "CPU"):
            lista_y = MC_Simulation(array_job_rank, MCsteps, 1, list_pairs_local_ham, eigenvectors[:,0], ee_list, Nsites, dt, R, T_grid, lista_y)

        
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(lista_y, MC_data_final, op=MPI.SUM, root=0)
        end = time.time()

        comm.reduce(cumulative_time_apply_local_gate, op=MPI.MAX, root=0)
        comm.reduce(cumulative_time_renyi, op=MPI.MAX, root=0)

        if (rank == 0):
            total_time = end - start
            print_times(cumulative_time_apply_local_gate, cumulative_time_renyi, total_time)

            with open('output.csv', 'a') as f:
                sys.stdout = f
                
                if (args.mode == "batchedGEMM"):
                    procedure_size_per_mpi = args.bs

                else:
                    procedure_size_per_mpi = size

                print_times_csv_format( args,
                                        procedure_size_per_mpi,
                                        cumulative_time_apply_local_gate,
                                        cumulative_time_renyi,
                                        total_time)

            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()
            

    if ( args.resume):
        #check for saved state folder
        if(not os.path.exists(states_dir) ):
            if(rank == 0):
                print("Error: no saved states, try without flag --resume.", flush=True)
                comm.Abort()
        comm.Barrier()
        
        number_of_saved_states = len(os.listdir(states_dir))
        
        # TODO safe check for number of states 
        #if( number_of_saved_states != size ):
        #    if( rank ==0 ):
        #        print("Error: number of parallel simulations is different than number of saved states. Try with mpirun -n", number_of_saved_states)
        #        print("or set the arguments to calculate the simulations from the beginning (without --resume)", flush=True)
        #        comm.Abort()
        #comm.Barrier()
        
        array_job_rank = rank + size * args.array_job_id

        if (args.mode == "batchedGEMM"):
            eigenvectors = []
            ee_list      = []
            
            for batch_num in range(batch_size):
                state_num = array_job_rank + batch_num
                saved_state_file = open(str(states_dir)+"/state_"+str(state_num), "rb")
                
                eigenvectors.append(pickle.load( saved_state_file) )
                ee_list.append(pickle.load( saved_state_file) )
                lista_y.append(pickle.load( saved_state_file))
                
                saved_state_file.close()
                
                MCsteps_old = len(lista_y[batch_num])
                MCsteps_to_cont = MCsteps - MCsteps_old

                if( MCsteps_to_cont < 10):
                    if( rank ==0 ):
                        print("Insert bigger number of monte carlo steps (MC) to resume simulations >", MCsteps_old+10)
                    comm.Barrier()
                    print("Current MC state of simulation ", state_num, " is: ", MCsteps_old)
                    comm.Abort()
                print("rank ", rank, " loaded state : " , state_num, "with steps: ", len(lista_y[batch_num]), ", number of steps to continue: ", MCsteps_to_cont, flush=True)
            comm.Barrier()
            eigenvectors = np.array(eigenvectors)
            
        else:    
            saved_state_file = open(str(states_dir)+"/state_"+str(array_job_rank), "rb")
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
                print("Current MC state of simulation ", array_job_rank, " is: ", MCsteps_old)
                comm.Abort()
            comm.Barrier()

            print("rank ", rank, " loaded state : " , array_job_rank, "with steps: ", lista_y.shape[0], ", number of steps to continue: ", MCsteps_to_cont, flush=True)

        list_pairs_local_ham = dep.generate_dictionary(Nsites,1, True)
        start = time.time()
        if( (args.mode == "GPU") or (args.mode == "batchedGEMM")):
            node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
            node_rank = node_comm.Get_rank()
            node_size = node_comm.Get_size()
            number_of_device_on_node = cp.cuda.runtime.getDeviceCount()

            # set device id to node ranks
            device_id  = node_rank % number_of_device_on_node 
            
            # set device to use 
            with cp.cuda.Device(device_id):
                
                dep.set_streams_global(batch_size)

                start_gpu = cp.cuda.Event()
                end_gpu = cp.cuda.Event()

                print("node rank", node_rank, "Starting simulation on device", cp.cuda.runtime.getDeviceProperties(device_id)['name'] ,flush=True)
               
                if(args.mode == "GPU"):
                    lista_y = MC_Simulation_GPU(array_job_rank, MCsteps, MCsteps_old, list_pairs_local_ham, eigenvectors, ee_list, Nsites, dt, R, T_grid, lista_y)           
               
                elif(args.mode == "batchedGEMM"):
                    lista_y = MC_Simulation_GPU_batch(array_job_rank, batch_size, MCsteps, MCsteps_old, list_pairs_local_ham, eigenvectors, ee_list, Nsites, dt, R, T_grid, lista_y)
           
                    #average by number of batch size
                    lista_y = np.average(lista_y, axis=0)

        if ( args.mode == "CPU"):
            lista_y = MC_Simulation(array_job_rank, MCsteps, MCsteps_old, list_pairs_local_ham, eigenvectors, ee_list, Nsites, dt, R, T_grid, lista_y)
             
        if( rank ==0 ):
            MC_data_final = np.empty(len(lista_y), dtype="float64")
        else:
            MC_data_final = None

        comm.Reduce(lista_y, MC_data_final, op=MPI.SUM, root=0)
        end = time.time()

        comm.reduce(cumulative_time_apply_local_gate, op=MPI.MAX, root=0)
        comm.reduce(cumulative_time_renyi, op=MPI.MAX, root=0)
    
        if (rank == 0):
            total_time = end - start
            print_times(cumulative_time_apply_local_gate, cumulative_time_renyi, total_time)

            with open('output.csv', 'a') as f:
                sys.stdout = f
                
                if (args.mode == "batchedGEMM"):
                    procedure_size_per_mpi = args.bs

                else:
                    procedure_size_per_mpi = size

                print_times_csv_format( args,
                                        procedure_size_per_mpi,
                                        cumulative_time_apply_local_gate,
                                        cumulative_time_renyi,
                                        total_time)

            save_file = open(filename.format(Nsites, R, lambdaa, MCsteps, size), "wb")
            pickle.dump(MC_data_final/size, save_file)
            save_file.close()
