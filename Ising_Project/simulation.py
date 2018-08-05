import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
import importlib
import constants
importlib.reload(constants)
from constants import *
import functions
importlib.reload(functions)
from functions import *

       
def start():
    if algorithm == "checkerboard":
        Num_time = Num_MCS*2
    elif algorithm == "metropolis":
        Num_time = Num_MCS*N**2
    elif algorithm == "wolff":
        Num_time = Num_MCS*N**2
    else:
        sys.exit("This is no valid algorithm: please enter 'metropolis' or 'wolff' or 'checkerboard'")
        
    
    start  = time.time()
    Tarray = np.linspace(T_sim,T_sim+T_range,NT)
   
    #initializing  
    cv     = np.zeros((len(Tarray),))
    errcv  = np.zeros((len(Tarray),))
    chi    = np.zeros((len(Tarray),))
    errchi = np.zeros((len(Tarray),))
    t = np.zeros((Num_time,))         #time in monte-carlo time steps
    m = np.zeros((len(Tarray),))
    errm = np.zeros((len(Tarray),))
    E_cor = np.zeros((2,Numtimecor))

    #loop over different temperatures
    for i in range(len(Tarray)):
        T = Tarray[i]
        spins = initialize_lattice(N,configuration)

        #initializing  
        E_lattice = np.zeros((Num_time,))
        m_lattice = np.zeros((Num_time,))
        MCS_wolff = np.zeros((Num_time,))
        m_lattice[0] = magnetisation(spins)
        E_lattice[0] = energy_ising(spins,J,H)
        timewolff = 0                             #initializing timestep for the wolff-algorithm
        MCS_count = 0                             #initializing the MCS-time for the wolff algorithm
        E_cor = np.zeros((2,Numtimecor))

        
        #simulations of ising model
        if algorithm == "metropolis":
            if heatbath == "on":
                for timestep in range(Num_time-1):
                    #Time evolution of the spins 
                    spins, delta_E, delta_m, flipped_spins = metropolis_heatbath(spins,T,J,H)
                    E_lattice[timestep+1] = E_lattice[timestep] + delta_E
                    m_lattice[timestep+1] = m_lattice[timestep] + delta_m
            elif heatbath == "off":
                for timestep in range(Num_time-1):
                    #Time evolution of the spins 
                    spins, delta_E, delta_m, flipped_spins = metropolis(spins,T,J,H)
                    E_lattice[timestep+1] = E_lattice[timestep] + delta_E
                    m_lattice[timestep+1] = m_lattice[timestep] + delta_m
                    
            else:
                sys.exit("heatbath must be specified to be on or off")
                
            #calculation thermodynamic properties, by using the stable part of the data    
            cv[i], errcv[i] = bootstrap_function(E_lattice[round(0.5*len(E_lattice)):],specific_heat, T, N)
            chi[i], errchi[i] = bootstrap_function(m_lattice[int(0.5*len(m_lattice)):], susceptibility, T, N)
            m[i],errm[i] = bootstrap_function(m_lattice[int(0.5*len(m_lattice)):], np.mean)
            E_cor = np.zeros((2,Numtimecor*N))
            t = np.linspace(0,Num_time-1,Num_time)/N**2        #time in monte-carlo timesteps
            E_cor[0,:],E_cor[1,:] = autocorrelation(E_lattice[round(0.5*Num_time):Num_time],t[round(0.5*Num_time):Num_time],Numtimecor*N)
            print('Progress: '+str(i/len(Tarray)*100)+'%', end="\r")

        elif algorithm == "wolff":
            #Because the number of MCS is not predefined for wolff, we need a while loop to check the progress
            while MCS_count<Num_MCS:
                #Time evolution of the spins 
                y_bonds,x_bonds        = computing_bonds(spins,J,T,kb=1)
                spins, n_cluster       = wolff(y_bonds,x_bonds,spins)
                m_lattice[timewolff+1] = magnetisation(spins)
                E_lattice[timewolff+1] = energy_ising(spins,J,H)
                
                #Determine monte carlo time steps
                MCS_count              = MCS_count + n_cluster/N**2
                MCS_wolff[timewolff+1] = MCS_count
                timewolff              = timewolff + 1
            
            #calculation thermodynamic properties, by using the stable part of the data     
            cv[i], errcv[i] = bootstrap_function(E_lattice[round(0.5*timewolff+1):timewolff+1], specific_heat, T, N)
            chi[i], errchi[i] = bootstrap_function(m_lattice[round(0.5*timewolff+1):timewolff+1],susceptibility,T,N)
            m[i],errm[i] = bootstrap_function(abs(m_lattice[round(0.5*timewolff+1):timewolff+1]),np.mean)
            

            E_cor[0,:],E_cor[1,:] = autocorrelation(E_lattice[round(0.5*timewolff+1):timewolff+1],MCS_wolff[round(0.5*timewolff+1):timewolff+1],Numtimecor)
            print('Progress: '+str(i/len(Tarray)*100)+'%', end="\r")


        elif algorithm == "checkerboard":
            if N%2 != 0:
                sys.exit("when using the checkerboard algorithm, the length of the lattice should be even")
            checker_board = make_checkerboard(N)
            
            for timestep in range(Num_time-1):
                #Time evolution of the spins 
                spins, delta_E        = checkerboard(spins,timestep,checker_board,T,J,H,kb=1)
                E_lattice[timestep+1] = E_lattice[timestep] + delta_E
                m_lattice[timestep+1] = magnetisation(spins)

            #calculation thermodynamic properties, by using the stable part of the data    
            cv[i], errcv[i] = bootstrap_function(E_lattice[round(0.5*len(E_lattice)):],specific_heat, T, N)
            chi[i], errchi[i] = bootstrap_function(m_lattice[int(0.5*len(m_lattice)):], susceptibility, T, N)
            m[i],errm[i] = bootstrap_function(m_lattice[int(0.5*len(m_lattice)):], np.mean)
            t = np.linspace(0,Num_time-1,Num_time)/2         #time in monte-carlo timesteps

            E_cor[0,:],E_cor[1,:] = autocorrelation(E_lattice[round(0.5*Num_time):Num_time],t[round(0.5*Num_time):Num_time],Numtimecor)
            print('Progress: '+str(i/len(Tarray)*100)+'%', end="\r")
      
    #calculate total time duration of simulation    
    stop = time.time()
    dt   = stop-start
    print("Time elapsed", dt)


    return t,E_lattice,m_lattice,spins,cv,chi,dt,Tarray,MCS_wolff,timewolff,m,errm,E_cor,errcv,errchi
