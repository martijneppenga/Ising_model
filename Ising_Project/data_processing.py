import numpy as np
import matplotlib.pyplot as plt
import datetime
from constants import *
def start(t,E_lattice,m_lattice,spins,cv,chi,dt,Tarray, MCS_wolff,timewolff,m,errm,E_cor,errcv,errchi):
    
    
    #for plot font
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=16)

    if NT>2:
        plt.errorbar(Tarray,cv,yerr=errcv, fmt = 'x')
        plt.xlabel('$k_b T/J$')
        plt.ylabel('$C_v$')
        plt.title('Specific heat')
        plt.show()

        plt.errorbar(Tarray,chi,yerr=errchi, fmt = 'x', label = 'Checkerboard')
        plt.xlabel('$k_b T/J$')
        plt.ylabel('$\chi$')
        plt.title('susceptibility')
        plt.show()
       
        
        plt.errorbar(Tarray,m, yerr=errm,fmt='x')
        plt.xlabel('$k_b T/J$')
        plt.ylabel('$m$')
        plt.title('Magnetisation')
        plt.show()
    
    
    if algorithm == "wolff":
        plt.plot(MCS_wolff[0:timewolff+1],(m_lattice[0:timewolff+1]))
        plt.xlabel('time (MCS)')
        plt.ylabel('Magnetisation (absolute value)')
        plt.title('Magnetisation')
        plt.show()
        
        plt.plot(MCS_wolff[0:timewolff+1],E_lattice[0:timewolff+1])
        plt.xlabel('time (MCS)')
        plt.ylabel('Lattice energy')
        plt.title('Energy')
        plt.show()        
        
    
    else:
        
        plt.plot(t,m_lattice)
        plt.title("Magnetisation vs time")
        plt.xlabel('time (MCS)')
        plt.ylabel('Magnetisation (absolute value)')
        plt.show()

        plt.plot(t,E_lattice)
        plt.title("Lattice energy vs time")
        plt.xlabel('time (MCS)')
        plt.ylabel('energy')
        plt.show()

    plt.plot(E_cor[1,:],E_cor[0,:])
    plt.title('Autocorrelation function')
    plt.xlabel('Time (MCS)')
    plt.ylabel('Function value')
    plt.show()