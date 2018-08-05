import numpy as np
#Physical constants of the system
#--------------------------------
J = 1                            #Interaction constant
H = 0                            #External magnetic field
kb = 1                           #Boltzmann constant (=1, because we want to normalize to kb)


#Input parameters for the simulation
#-----------------------------------
heatbath = "on"                 #Use of heatbath algorithm for metropolis
algorithm = "metropolis"        #enter "metropolis" or "wolff" or "checkerboard"
configuration = "hot"           #enter the initial configuration of the spins choose from: 'hot', 'cold', 'random', 'spin_up',  'spin_down'
N = 20                          #Lattice size of (NxN) spins
Num_MCS = 10                    #Number of MCS
T_sim = 0.1                     #Temperature of the simulation
T_range = 5                     #Temperature range in case a range of temperatures should be evaluated
NT = 10                         #Number of T-values evaluate in one simulation
Numtimecor = 2                  #Number of samples for autocorrelation function

#Input parameters for the animation
#----------------------------------
delay_time =  10                #delay between frames in milliseconds
