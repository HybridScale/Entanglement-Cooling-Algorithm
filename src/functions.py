import numpy as np
import cupy as cp
from mpi4py import MPI
import pickle
import os
import dependecies as dep
from scipy.sparse.linalg import LinearOperator, eigsh, eigs
import time
import sys



def CPUIteration(self, simID, cc, MCsteps_old, print_exponent, loc_pairs, state, y, ee_list_old):
    accepted = 0

    MCsteps        = self.MC
    MCsteps_wanted = self.configuration["MCwanted"]  
    
    Nsites = self.configuration["Nsites"]
    R      = self.configuration["R"]
    Lambda = self.configuration["lambda"]
    dt     = self.dt
    T_grid = self.T_grid

    for step in range(MCsteps_old, MCsteps+1):
        random_pair, random_pair_id = dep.select_cooling_evolution_indices(loc_pairs)
        random_gate, random_gate_id = dep.select_cooling_evolution_gates()

        dep.timing.Start("ApplyLocalGate")
        new_state = dep.ApplyLocalGate(random_gate,random_pair,state, Nsites, 2, dt)
        dep.timing.Stop("ApplyLocalGate")

        dep.timing.Start("RenyEntropy")
        ent_new, ee_list_new, relevant_partitions = dep.Renyi2_aftergate_correct(Nsites, R, \
                                                                                 new_state,random_pair)
        dep.timing.Stop("RenyEntropy")

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

        y = np.append(y, np.average(ee_list_old))

        if ( MCsteps >=100 and step % int(MCsteps_wanted/100) == 0):
            cc += 1

        if( step %  print_exponent == 0 ):
            print(f"rank {simID} MC step = {step}")

        if( step % 1_000 == 0 and MCsteps > MCsteps_old):
            with open(self.states_dir + f"/state_{simID}", "wb+") as f:
                pickle.dump(state, f)
                pickle.dump(ee_list_old, f)
                pickle.dump(y, f)
                pickle.dump(cc, f)
            print(f"rank {simID} saved state on step {step}")

    return y, accepted

def GPUIteration(self, simID, cc, MCsteps_old, print_exponent, loc_pairs, state, y, ee_list_old):
    accepted = 0
    state_DEVICE = cp.asarray(state)
    y = cp.asarray(y)

    MCsteps = self.MC
    MCsteps_wanted = self.configuration["MCwanted"]

    Nsites = self.configuration["Nsites"]
    R      = self.configuration["R"]
    Lambda = self.configuration["lambda"]
    dt     = self.dt
    T_grid = self.T_grid

    # Create and return sigma gates on device (cupy array)
    sigma_list_device = dep.create_sigma_list_GPU(dt)
    
    rand_sigma_value = len(sigma_list_device)
    
    for step in range(MCsteps_old, MCsteps+1):
        random_pair, random_pair_id = dep.select_cooling_evolution_indices(loc_pairs)
        random_sigma_gate = sigma_list_device[np.random.randint(rand_sigma_value)]

        dep.timing.Start("ApplyLocalGate")
        new_state_DEVICE = dep.ApplyLocalGate_GPU(random_sigma_gate, random_pair, state_DEVICE, \
                                                  Nsites, 2, dt)
        dep.timing.Stop("ApplyLocalGate")

        dep.timing.Start("RenyEntropy")
        ent_new_DEVICE, ee_list_new_DEVICE, relevant_partitions = dep.Renyi2_aftergate_correct_GPU(Nsites, R, \
                                                                                                   new_state_DEVICE,random_pair)
        dep.timing.Stop("RenyEntropy")
        
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

        y = np.append(y, np.average(ee_list_old))

        if( step %  print_exponent == 0 ):
            print(f"rank {simID} MC step = {step}")

        if ( MCsteps >=100 and step % int(MCsteps_wanted/100) == 0):
            cc += 1

        if( step % 1000 == 0 and MCsteps>MCsteps_old):
            with open(self.states_dir + f"/state_{simID}", "wb+") as f:
                pickle.dump(state_DEVICE, f)
                pickle.dump(ee_list_old, f)
                pickle.dump(y, f)
                pickle.dump(cc, f)
            print(f"rank {simID} saved state on step {step}")
   
    return cp.asnumpy(y), accepted


def batchGEMMIteration(self, simID, cc, MCsteps_old, print_exponent, loc_pairs , state, y, ee_list_old):
    accepted = 0

    MCsteps        = self.MC
    MCsteps_wanted = self.configuration["MCwanted"]
    batch_size     = self.batch_size

    Nsites = self.configuration["Nsites"]
    R      = self.configuration["R"]
    Lambda = self.configuration["lambda"]
    dt     = self.dt
    T_grid = self.T_grid

    state       = cp.tile(state, (batch_size, 1))
    ee_list_old = cp.tile(ee_list_old, (batch_size, 1))

    dep.set_streams_global(batch_size)
    
    sigma_list_device = dep.create_sigma_list_GPU(dt)
    
    for step in range(MCsteps_old, MCsteps+1):
        random_pairs, random_pair_ids = dep.select_cooling_evolution_indices_batch(loc_pairs, batch_size)

        random_sigma_gates = dep.select_sigma_gates_batch(sigma_list_device, batch_size)

        new_states = dep.ApplyLocalGate_GPU_batch(random_sigma_gates, 
                                                  batch_size ,
                                                  random_pairs, 
                                                  state, 
                                                  Nsites, 
                                                  2, 
                                                  dt)

        ents_new, ees_list_new, relevant_partitions = dep.Renyi2_aftergate_correct_GPU_batch(Nsites, 
                                                                                             R, 
                                                                                             new_states, 
                                                                                             batch_size,
                                                                                             random_pairs)

        ent_tamp = ee_list_old.copy()
        for batch_num in range(batch_size):             

            ent_tamp[batch_num] [relevant_partitions[batch_num][0]] = ees_list_new[batch_num][0]
            ent_tamp[batch_num] [relevant_partitions[batch_num][1]] = ees_list_new[batch_num][1]

        old_value = np.average( ee_list_old, axis = 1)
        new_value = np.average( ent_tamp, axis = 1)
            
        for batch_num in range(batch_size): 
            p_exp = -(1.0/T_grid[cc])*(new_value[batch_num] - old_value[batch_num])
            if (p_exp > 50):
                p = 1.0
            else:
                p = np.exp(p_exp)

            random_value = np.random.uniform(0,1)
            if (random_value <= min(1.0,p)):
                accepted += 1
                state[batch_num] = new_states[batch_num].copy()
                ee_list_old[batch_num] = ent_tamp[batch_num].copy()
        
        if (step == 1):
            y = np.average(ee_list_old, axis = 1).reshape(batch_size, 1)
        else:
            y = np.append(y, np.average(ee_list_old, axis = 1).reshape(batch_size, 1), axis = 1)
            
        if( step %  print_exponent == 0 ):
            print(f"rank {simID} MC step = {step}")
            
        if (step % int(MCsteps/100) == 0):
            cc += 1

        if( step % 1000 == 0 and MCsteps>MCsteps_old):
            for batch_num in range(batch_size):
                with open(self.states_dir + f"/state_{simID}", "wb+") as f:
                    pickle.dump(state[batch_num].get(), f)
                    pickle.dump(ee_list_old[batch_num], f)
                    pickle.dump(y[batch_num], f)
            print(f"rank {simID} saved state on step {step}")
            
    return cp.asnumpy(y)