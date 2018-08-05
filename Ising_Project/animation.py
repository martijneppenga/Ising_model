import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider
import sys
import matplotlib.patches as mpatches
from functions import *
from constants import *

# Initialise figure
spins = initialize_lattice(N,configuration)
fig = plt.figure()
ax = fig.add_subplot(111)
im = plt.imshow(spins, cmap='gist_gray_r', vmin=0, vmax=1)

# Define an axes area and draw a slider in it
T_slider_ax  = fig.add_axes([0.25, 0.15, 0.65, 0.03] )
T_slider = Slider(T_slider_ax, 'T', 0, 10.0, valinit=T_sim)
H_slider_ax  = fig.add_axes([0.25, 0.05, 0.65, 0.03] )
H_slider = Slider(H_slider_ax, 'H', -5, 5, valinit=0)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(bottom=0.25)

def init():
    im.set_data(spins)
timestep=0

# define function to make animation for each algorithm
if algorithm == "metropolis":
    if heatbath == "on":
        
        def animate(timestep):
            global spins
            H = H_slider.val
            T = T_slider.val
            for i in range(50):
                spins, delta_E, delta_m, flipped_spins = metropolis_heatbath(spins,T,J,H)

            im.set_data(spins)
            return im
    elif heatbath == "off":
      
        def animate(timestep):
            global spins
            H = H_slider.val
            T = T_slider.val
            for i in range(50):
                spins, delta_E, delta_m, flipped_spins = metropolis(spins,T,J,H)
            im.set_data(spins)
            return im

    else:
        sys.exit("heatbath must be specified to be on or off")
        
        
elif algorithm == "checkerboard":
    checker_board = make_checkerboard(N)
    def animate(timestep):
        global spins
        H = H_slider.val
        T = T_slider.val
        spins, delta_E = checkerboard(spins,timestep,checker_board,T,J,H,kb=1)
        im.set_data(spins)
        return im

elif algorithm == "wolff":    
    def animate(timestep):
        global spins
        H = H_slider.val
        T = T_slider.val
        for i in range(2):
            y_bonds,x_bonds   = computing_bonds(spins,J,T,kb=1)
            spins,num_cluster = wolff(y_bonds,x_bonds,spins)
        im.set_data(spins)
        return im
else:
    sys.exit("This is no valid algorithm: please enter 'metropolis' or 'wolff' or 'checkerboard'")




anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=delay_time, blit=True)
plt.show()