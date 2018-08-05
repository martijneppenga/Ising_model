# Monte Carlo simulation of the 2 dimensional Ising model 

This second project for the computational physics course comprises the Monte Carlo simulation of the Ising model. This simulation uses the Metropolis, Checker-board or Wolff algorithm on a square lattice with periodic boundary conditions to simulate (anti)-ferromagnetism. 
Thermodynamic quantities such as magnetisation, susceptibility and specific heat are obtained from this simulation.

## Getting Started
In order to run the simulation, copy all the .py files in the repository to your working directory. Change the input constants in constants.py, save the file, then start the simulation from the simulation module, and use the results to do the processing, as follows:

```
import importlib
import simulation
import data_processing

importlib.reload(simulation)
results = simulation.start()
importlib.reload(data_processing)
data_processing.start(*resultswolff)
```

Or, simply run launcher.py which does all of the above steps.

An animation of the simulation can be made with the animation.py file. Change the input constants in constants.py, save the file, then start the animation from the animation module as follows: 
```
%matplotlib notebook
import animation
```

During the animation the temperature and external magnetic field of the lattice can be change with the sliders underneath the animation.
### Options for constants

The following inputs can be adjusted in constants.py

Constants related to Hamiltonian of the Ising model
* **J**  Interaction constant
* **H**  External magnetic field
* **kb** Boltzmann constant (leave at one to use dimensionless units)


Parameters of the system
* **heatbath**      Use of heatbath algorithm for metropolis. Options are 'on' and 'off')
* **algorithm**     Choice of which algorithm the simulations uses. Options are:  "metropolis" or "wolff" or "checkerboard"
* **configuration** Initial configuration of the spins options are: 'hot' (each spin random up or down), 'cold' (all spins up), 'random' (each spin random up or down), 'spin_up' (all spins up),  'spin_down' (all spins down)
* **N**             Lattice size of (NxN) spins. For Wolff algorithm N should be N<=40 and for the checkerboard algorithm N should be even.
* **Num_MCS**       Length of simulation in number of Monte Carlo time steps. Each Monte Carlo time step means that, on average, all spins are flipped once
* **T_sim**         The temperature of the system
* **T_range**       Temperature range in case a range of temperatures should be evaluated (range is from T_sim to T_sim+T_range)
* **NT**            Number of Temperature-values evaluate in one simulation
* **Numtimecor**    Number of samples for autocorrelation function

Parameters for the animation
* **Delay_time** Delay between frames in milliseconds


### Obtained results
The output of the simulation contains:
* **t**         Time in Monte Carlo steps for the metropolis or checkerboard algorithm
* **E_lattice** Energy of the lattice for all time steps for last evaluated temperature
* **m_lattice** Magnetization of the lattice for all time steps evaluated temperature
* **spins**     Final configuration for the spins
* **cv**        Specific heat for all temperatures
* **chi**       Susceptibility for all the time steps
* **dt**        Time duration of the simulation
* **Tarray**    Evaluated temperatures during the simulation
* **MCS_wolff** Time in Monte Carlo steps for the wolff algorithm
* **timewolff** Number of loops need with the wolff algorithm
* **m**         Average magnetization for all temperatures
* **errm**      Standard deviation of the average magnetization
* **E_cor**     Auto-correlation function with corresponding Monte Carlo timescale
* **errcv**     Standard deviation of the specific heat
* **errchi**    Standard deviation of the Susceptibility





## Authors
* Martijn Eppenga
* Richard Faasse
* Cyrus Tirband 

## Report
The report for this project can be found in the repository as well.