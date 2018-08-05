import numpy as np
import sys


def bootstrap_function(data, function, *func_args, n=1000):
    """bootstrap_function(data, function n,*func_args): 
    Description:
    ------------
    This function takes an array of data, and a function (such as np.mean, if     
    the quantity is a mean of the data), together with its function arguments to produce 
    The bootstrap function then returns the quantity and its standard deviation
    
    Parameters:
    ----------
    data: array (can be any size)
        Data from which the quantity needs to be calculated
    function: function
        Function which calculates the quantity
    func_args: can be any format, depend on the parameters of the function
        Arguments that are needed for the input function
    n: float
        Number of random sample that are picked from the data
        Default: n = 1000
    Results:
    --------
    quantity: can be anything
        The quantity calculated with the function and data
    std_quantity: float
        Standard deviation of the quantity
    """
    
    N = len(data)
    sample_ind = np.floor(np.random.rand(N,n)*N).astype(int)
    #take random samples of the data
    data_resample = data[sample_ind]

    #compute the quantity and its standard deviation
    quantity_resample = function(data_resample, *func_args, axis=0)
    std_quantity = np.std(quantity_resample)
    quantity = function(data, *func_args, axis=0)
    return quantity, std_quantity


def shift_lattice(lattice,direction):
    """shift_lattice(lattice,direction):
    Description:
    ------------
    This algorithm shift a the rows or columns of a matrix to the left, right, up or down and re-introduces the last shifted row or column
    at the first row or column.
    
    Parameters:
    -----------
    lattice: array of size(N,M)
        input array to shift
    direction: string 
        direction in which the matrix is shifted. Options are: left, right, up, down
    
    Results:
    --------
    matrix_shift: array of size (N,M)
        The shifted matrix
    """
    size_lattice = np.shape(lattice)
    
    # initialize matrix for shift copies of the lattice
    matrix_shift = np.zeros(size_lattice)
   
    
    if direction == 'down':
        # shift the lattice one row down
        matrix_shift[1:,:] = lattice[0:-1,:]
        matrix_shift[0,:] = lattice[size_lattice[1]-1,:]
        
    if direction == 'up':
        # shift the lattice one row up
        matrix_shift[0:-1,:] = lattice[1:,:]
        matrix_shift[size_lattice[1]-1,:] = lattice[0,:]
    
    if direction == 'left':
        # shift the lattice one column to the left
        matrix_shift[:,0:-1] = lattice[:,1:]
        matrix_shift[:,size_lattice[1]-1] = lattice[:,0]
    
    if direction == 'right':
        # shift the lattice one column to the right
        matrix_shift[:,1:] = lattice[:,0:-1]
        matrix_shift[:,0] = lattice[:,size_lattice[1]-1]
    
    return matrix_shift




def energy_ising(lattice,J,H=0):
    """energy_ising(lattice,J,H=0)
    Description:
    ------------
    This function calculates the energy of Ising model on a square lattice using E = sum_{i,j} -J*S_i*S_j-H*s_i
    where i,j are the nearest neighbor pairs (four for a square lattice)
    The function assumes that the lattice has periodic boundary conditions
    
    Parameter:
    ----------
    lattice: array size of (N,N)
        Array containing the spin value of each spin on a lattice site
    J: float
        Interaction constant
    H: float
     
     External magnetic field
        default: H = 0
    
    Results:
    --------
    energy: float
        Energy of Ising model configuration 
        
    """
    
    # Determine for each spin the sum of parallel neighbor spins minus the number antiparallel neighbor spins. And multiply the anwser 
    # with 0.5 to avoud double counting for the energy calculation
    marrix_shift_up    = shift_lattice(lattice,'up')
    matrix_shift_down  = shift_lattice(lattice,'down')
    matrix_shift_left  = shift_lattice(lattice,'left')
    matrix_shift_right = shift_lattice(lattice,'right')
    spin_spin_int      = 0.5*(matrix_shift_down+marrix_shift_up+matrix_shift_left+matrix_shift_right)*lattice 
    
    #calculate total energy of the lattice 
    if H != 0:
        #energy with magnetic field
        energy = -J*np.sum(spin_spin_int)-H*np.sum(lattice)
        
    elif H == 0:
        #energy without magnetic field
        energy = -J*np.sum(spin_spin_int)
        
    return energy


def magnetisation(lattice):
    """magnetisation(lattice)
    Description:
    ------------
    This function calculates the mean magnetization by taking the mean value of the lattice containing the individual spins.
    
    Parameter:
    ----------
    lattice: array of size (N,N)
        Array containing the spin value of each spin on a lattice site

    Results:
    --------
    m: float
        mean magnetization  
        
    """
    m = np.mean(lattice)
    
    return m

def specific_heat(energy, T, N, axis=0):
    """specific_heat(energy, T, N):
    Description:
    ------------
    This function calculates the specific heat by calculating the variance in the energy-array
    
    Parameters:
    -----------
    energy: array of size (Num_time,1)
        Total energy of the spin-lattice at each time step
    T: float
        temperature 
    N: float
        Number of spins in one dimension
    
    Results:
    --------
    cv: float
        Specific heat at a specific temperature
    """
    
    cv = 1/(N**2*T**2)*np.var(energy,axis=0)
    
    return cv

def susceptibility(magnetization, T, N, axis = 0):
    """susceptibility(magnetization, T, N):
    Description:
    ------------
    This function calculates the susceptibility by calculating the variance in the magnetization-array
    
    Parameters:
    -----------
    magnetization: array of size (Num_time,1)
        mean magnetization of the spin-lattice at each timestep
    T: float
        temperature 
    N: float
        Number of spins in one dimension
          
    
    Results:
    --------
    chi: float
        Susceptibility at a specific temperature
        
    """
    
    chi = N**2/T*np.var(abs(magnetization),axis=0)
    

    return chi

def initialize_lattice(N,configuration):
    """initialize_lattice(N,configuration):
    Description:
    ------------
    This function makes a square lattice with spins up or down
    
    Parameter:
    ----------
    N: float
        Number of spins along x and y axis (total number of spins is N^2)
    configuration: string
        Configuration of lattice: choose from: 'hot', 'cold', 'random', 'spin_up', 'spin_down'
            'hot' and 'random' generates random distrusted spins
            'cold' and 'spin_up' generates lattice with all spins up
            'spin_down' generates lattice with all spins down
            
    Results: array of size (N,N)
        array containing the value of each spin on the lattice
    
    
    
    """
    name = 'false'
    while name == 'false':
        if configuration == 'random' or configuration == 'hot':
            spins = 2*np.round(np.random.rand(N,N))-1
            name  = 'true'
        elif configuration == 'spin_up' or configuration == 'cold':
            spins = np.ones((N,N))
            name  = 'true'
        elif configuration == 'spin_down':
            spins = -np.ones((N,N))
            name  = 'true'
        else:
            print('The desired lattice configuration is not supported')
            print('Please choose one of the following supported lattice configurations:')
            print('random, hot, cold, spin_up, spin_down')
            begin = str(input('Lattice configuration: '))
                
    return spins

def metropolis_heatbath(spins,T,J,H,kb=1):
    """spin_flip(spins,T,J,kb=1):
    Description:
    -----------
    This function chooses at random a spin from a given lattice and decides if it will flip the spin or not.
    It makes to choose of flipping a spin by calculating the change in energy of the lattice before and after the spins flip.
    The energy is calculated with the nearest neighbors (horizontal and vertical) interaction: E = sum_{i,j} -J*s_i*s_j  
    The following criteria is used to determine if a spin should be flipped or not:
    If delta_E > 0: flip spin with probability exp(-delta_E/(kb*T))
    If delta_E < 0: Always flip spin
    where delta_E is the difference in energy of lattice before and after flipping a spin)
    
    Parameters:
    -----------
    spins: array of size (N,N)
        Array containing the spin value of each spin on a lattice site
    T: float
        temperature 
    J: float
        Interaction constant
    kb: float
        boltzmann constant
        default: kb = 1
        
    Results:
    --------
    spins: array of size (N,N)
        Array containing the spin value of each spin on a lattice site with one spin flipped
    delta_E: float
        difference in energy of the lattice
    delta_m: float
        difference in mean magnetization of the lattice
    coordinates_flipped_spin: array of size (2,1)
        array containing the position of the flipped spin in the spins array
        first element is the row, second element is the column
    """
    
    # choose a spin particle from the lattice at random
    N     = len(spins)
    xrand = np.int_(np.floor(N*np.random.rand()))
    yrand = np.int_(np.floor(N*np.random.rand()))
    
    coordinates_flipped_spin =[xrand,yrand]
    current_spin = spins[xrand,yrand]
    
    #determine nearest neighbors 
    xneighbors = np.array([(xrand-1)%N,(xrand+1)%N,xrand,xrand])
    yneighbors = np.array([yrand,yrand,(yrand-1)%N,(yrand+1)%N])
    
    #compute energy before flipping
    E_before = -J*np.sum(spins[xneighbors,yneighbors]*current_spin) 
    delta_E  = -2*E_before + 2*H*current_spin
    
    #determine  probability of flipping spin
    rand   = np.random.rand()
    n_plus = np.sum(spins[xneighbors,yneighbors] == 1)
    p_plus = np.exp((2*n_plus-4)*J/(kb*T))/(np.exp((2*n_plus-4)*J/(kb*T))+np.exp(-(2*n_plus-4)*J/(kb*T)))

    #Set the spin in the up position if the random number is smaller than the probability to have an up spin,
    #otherwise set the spin in the down positon
    if (rand<p_plus):
        if (current_spin == 1):
            delta_E = 0
            delta_m = 0
        else: 
            delta_m = current_spin*2/N**2
        spins[xrand,yrand] = 1
    else:
        if (current_spin == -1):
            delta_E = 0
            delta_m = 0
        else: 
            delta_m = -current_spin*2/N**2
        spins[xrand,yrand] = -1


    return spins, delta_E, delta_m, coordinates_flipped_spin

def metropolis(spins,T,J,H,kb=1):
    """spin_flip(spins,T,J,kb=1):
    Description:
    -----------
    This function chooses at random a spin from a given lattice and decides if it will flip the spin or not.
    It makes to choose of flipping a spin by calculating the change in energy of the lattice before and after the spins flip.
    The energy is calculated with the nearest neighbors (horizontal and vertical) interaction: E = sum_{i,j} -J*s_i*s_j  
    The following criteria is used to determine if a spin should be flipped or not:
    If delta_E > 0: flip spin with probability exp(-delta_E/(kb*T))
    If delta_E < 0: Always flip spin
    where delta_E is the difference in energy of lattice before and after flipping a spin)
    
    Parameters:
    -----------
    spins: array of size (N,N)
        Array containing the spin value of each spin on a lattice site
    T: float
        temperature 
    J: float
        Interaction constant
    kb: float
        boltzmann constant
        default: kb = 1
        
    Results:
    --------
    spins: array of size (N,N)
        Array containing the spin value of each spin on a lattice site with one spin flipped
    delta_E: float
        difference in energy of the lattice
    delta_m: float
        difference in mean magnetization of the lattice
    coordinates_flipped_spin: array of size (2,1)
        array containing the position of the flipped spin in the spins array
        first element is the row, second element is the column
    """
    
    # choose a spin particle from the lattice at random
    N     = len(spins)
    xrand = np.int_(np.floor(N*np.random.rand()))
    yrand = np.int_(np.floor(N*np.random.rand()))
    
    coordinates_flipped_spin = [xrand,yrand]
    current_spin = spins[xrand,yrand]
    
    # Determine nearest neighbors 
    xneighbors = np.array([(xrand-1)%N,(xrand+1)%N,xrand,xrand])
    yneighbors = np.array([yrand,yrand,(yrand-1)%N,(yrand+1)%N])
    
    #compute energy before flipping
    E_before = -J*np.sum(spins[xneighbors,yneighbors]*current_spin) 
    delta_E  = -2*E_before + 2*H*current_spin
    
    # deterimne probability of flipping spin
    p_accept = np.exp(-delta_E/(kb*T))
    rand     = np.random.rand()
    
    
    if (rand<p_accept):
        # flip spin
        spins[xrand,yrand] = spins[xrand,yrand]*-1
        delta_m = spins[xrand,yrand]*2/N**2 
    else:
        # don't flip spin
        delta_E = 0
        delta_m = 0


    return spins, delta_E, delta_m, coordinates_flipped_spin



def computing_bonds(lattice,J,T,kb=1):
    """computing_bonds(lattice,J,T,kb=1):
    Description:
    -----------
    This function computes the bonds between the spins in a lattice of spins (needed for the Wolff method to form clusters) in the following way:
    It checks whether two neighboring spins are parallel or antiparallel:
      - If the spins are parallel:
          Freeze or remove a bond between spins with a probability according to the probability distribution exp(-2*J/kb/T)
      - If the spins are anti-parallel:
          remove bond between spins
    Checking all spins (in a vectorised way) gives the result of matrices with all locations of the bonds
    
    Parameters:
    -----------
    lattice: array of size (N,N)
        Array containing the spin value of each spin on a lattice site
    T: float
        temperature 
    J: float
        Interaction constant
    kb: float
        boltzmann constant
        default: kb = 1
        
    Results:
    --------
    xbonds: array of size (N,N)
        Array containing the information about the bonds, where the element (i,j) gives a 1 or 0 depending on if the spins at position (i,j) and position (i,j-1) have a frozen bond (1) or removed bond (0).
    ybonds: array of size (N,N)
        Array containing the information about the bonds, where the element (i,j) gives a 1 or 0 depending on if the spins at position (i,j) and position (i-1,j) have a frozen bond (1) or removed bond (0).
    """
    
    size_lattice = np.shape(lattice)
    
    
    # shift the lattice one row down and one column right 
    matrix_shift_down  = shift_lattice(lattice,'down')
    matrix_shift_right = shift_lattice(lattice,'right')

    # identifing bonds:
    y_bonds = matrix_shift_down*lattice
    x_bonds = matrix_shift_right*lattice
    
    y_bonds[y_bonds == -1] = 0
    x_bonds[x_bonds == -1] = 0
    
    y_bonds = y_bonds*np.random.rand(size_lattice[0],size_lattice[1])
    x_bonds = x_bonds*np.random.rand(size_lattice[0],size_lattice[1])
    
    probabillity = np.exp(-2*J/kb/T)
    
    # remove bond between spins
    y_bonds[y_bonds<probabillity] = 0
    x_bonds[x_bonds<probabillity] = 0
    
    # freeze bond betweens spins
    x_bonds[x_bonds>0] = 1
    y_bonds[y_bonds>0] = 1
    
    return y_bonds,x_bonds


def wolff(y_bonds,x_bonds,spins):
    """wolff(y_bonds,x_bonds,spins):
    Description:
    -----------
    This function groups the spins together in clusters with a recursive method and at the same time flips these clusters with a 50% chance. It returns the updated matrix containing the spin-lattice. It furthermore returns the number of spins in one cluster    
    
    Parameters:
    -----------
    xbonds: array of size (N,N)
        Array containing the information about the bonds, where the element (i,j) gives a 1 or 0 depending on if the spins at position (i,j) and position (i,j-1) have a bond (1) or not (0).
    ybonds: array of size (N,N)
        Array containing the information about the bonds, where the element (i,j) gives a 1 or 0 depending on if the spins at position (i,j) and position (i-1,j) have a bond (1) or not (0).
    spins: array of size (N,N)
        Array containing the spin value of each spin on a lattice site
    
        
    Results:
    --------
    spins: array of size (N,N)
        Array containing the updated spin value of each spin on a lattice site, in which clusters are randomly flipped
    n_cluster: float
        Number of spins in the cluster of the wolff time step
    """
    N = len(y_bonds)
    
    #matrix containing the information if a spins is visited (1), or not visited (0)
    visit_position = np.zeros((N,N))
   
    #backtrack algorithm. Recursive algorithm that identifies spin cluster. It moves over the entire lattice and keeps
    #track if spin position is visited (1) or not visited (0). Function is based on a backtrack function from the book computational
    #physics by J. M. Thijssen, chapter 15.5.1
    def back_track(i,j,spins,flip_spin):
        if visit_position[i,j] == 0:
            visit_position[i,j] = 1
            if flip_spin == 1:
                spins[i,j] = -spins[i,j]
                
            if x_bonds[i,(j+1)%N] == 1 :
                back_track(i,(j+1)%N,spins,flip_spin)
                
            if y_bonds[i,j] == 1:
                back_track((i-1)%N,j,spins,flip_spin)
                
            if x_bonds[i,j] == 1:
                back_track(i,(j-1)%N,spins,flip_spin) 
                
            if y_bonds[(i+1)%N,j] == 1: 
                 back_track((i+1)%N,j,spins,flip_spin)
                    
    
    #Flip the spins in a cluster with a 50% probability
    flip_spin = 1
    back_track(int(np.floor(N*np.random.rand())),int(np.floor(N*np.random.rand())),spins,flip_spin)
    
    #Determine number of spins in the cluster
    n_cluster = np.sum(visit_position)
    
    return spins, n_cluster





def checkerboard(spins,timestep,checker_board,T,J,H,kb=1):
    """checkerboard(spins,timestep,checker_board,T,J,H,kb=1):
    Description:
    -----------
    This function flips the spins of the spin configuration with periodic boundary conditions at checker board positions under the 
    following conditions:
    If a spin flip lowers the energy of the lattice, then the spin is always flipped
    If a spin flip increases the energy then the spin is flipped with a probability of exp(-delta_E/(Kb*T))
    If a spin flip does not change the energy, then the spin is not flipped
    
    If the time step input is odd then the first checkerboard positions are used. If the time step is odd then the second 
    checkerboard positions are used (for instance by an odd number the white positions of a checkerboard are used, and by an
    even number the black positions of a checkerboard are used)
    
    The parameter checker_board can be made with the function make_checkerboard
    
    
    Parameters:
    -----------
    spins: array of size (N,N)
        Array containing the spin value of each spin on a lattice site
    timestep: integer 
        odd or even number telling which positions on the checkerboard to use
    checker_board: array of size (2,N,N)
        array containing two (N,N) size matrices of alternating zeros and ones on each row and column. 
    T: float
        Temperature 
    J: float
        Interaction constant
    kb: float
        boltzmann constant
        default: kb = 1
        
    Results:
    --------
    spins: array of size (N,N)
        array containing the spins on the lattice
    delta_E: float
        difference in energy of the lattice
    """
    size_lattice = np.shape(spins)
    
    #determine energy of the spins
    marrix_shift_up    = shift_lattice(spins,'up')
    matrix_shift_down  = shift_lattice(spins,'down')
    matrix_shift_left  = shift_lattice(spins,'left')
    matrix_shift_right = shift_lattice(spins,'right')
    delta_E_spins      = 2*J*(matrix_shift_down+marrix_shift_up+matrix_shift_left+matrix_shift_right)*spins+2*H*spins

    #Create matrix to flip a spin (-1) or to not flip a spin(1)
    P_accept_spins = np.exp(-delta_E_spins/kb/T)
    flip_spins     = 2*(P_accept_spins>np.random.rand(size_lattice[0],size_lattice[0]))-1
    
    #Flip the spins at the checker_board positions according to flip_spins matrix and determine  total energy of the configuration
    spins   = (-1*flip_spins*checker_board[timestep%2,:,:]+checker_board[(timestep+1)%2,:,:])*spins
    delta_E = np.sum((flip_spins+1)*0.5*delta_E_spins*checker_board[timestep%2,:,:])
    
    return spins, delta_E

def make_checkerboard(N):
    """make_checkerboard(N):
    Description:
    -----------
    This functions makes a checkerboard lattice. The checkerboard lattice is a matrix of alternating zeros and ones on each row and column 
    
    Parameters:
    -----------
    N: float
        number of rows and columns of the checkerboard lattice
        
    Results:
    --------
    checker_board: Array of size (N,N)
        Array of alternating zeros and ones for each column and row
    
    
    """
    re = np.r_[ int(N/2)*[0,1] ]              # even-numbered rows
    ro = np.r_[ int(N/2)*[1,0] ]              # odd-numbered rows
    
    
    checker_board        = np.zeros((2,N,N))
    checker_board[0,:,:] = np.row_stack(int(N/2)*(re, ro))
    checker_board[1,:,:] = np.row_stack(int(N/2)*(ro, re))
    
    return checker_board

def autocorrelation(data,MCS,Numtimecor):
    """autocorrelation(data,MCS,Numtimecor):
    Description:
    -----------
    This functions computes the autocorrelation function from input data.
    
    Parameters:
    -----------
    data: array
        input data the autocorrelation function should be computed of
    MCS : array
        simulation time in MCS, corresponding to the input data
    Numtimecor: float
        Number of data points used in this function to compute the autocorrelation function
        
    Results:
    --------
    dataconv: Array of size (2*Numtimecor+1,) 
        Array containing the autocorrelation function
    MCS_scale: Array of size (2*Numtimecor+1,)
        Array containing the MCS-scale at which the autocorrelation function is computed.
    """
    
    if len(data)>Numtimecor:
        data_cor = data[0:Numtimecor]-np.mean(data[0:Numtimecor])
         
        dataconv = np.correlate(data_cor,data_cor,'full')
        dataconv = dataconv[Numtimecor-1:]/np.max(dataconv)
        MCS_scale = MCS[0:Numtimecor]
        MCS_scale = MCS_scale - MCS_scale[0]
        return dataconv, MCS_scale
    else:
        sys.exit('Sample is not long enough for computing the auto-correlation function')

def sampling(data,samplestep):
    """sampling(data,samplestep):
    Description:
    -----------
    This functions samples inputted data with a period of samplestep
    
    Parameters:
    -----------
    data: array
        input data that should be sampled
    samplestep: int
        period of the desired sampling

        
    Results:
    --------
    sampled_data: Array of size (floor(len(data)/samplestep),) 
        Array containing the sampled data

    """
    sampled_data = np.zeros(int(len(data)/samplestep),)
    for i in range(int(len(data)/samplestep)):
        sampled_data[i] = data[samplestep*i]
        
    return sampled_data



        