# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 19:03:08 2018

@author: malopez
"""
import numpy as np
from numpy import random_intel
import matplotlib.pyplot as plt
from functions import propagate, computeCollisions, printProgressBar, compute_a2, findMaximumRelativeVelocity
from graphics import plotTheoryVsSimulation
                    

# Simulation variables
m = 1
effective_diameter = 1
effective_radius = effective_diameter/2
# Total number of particles in the system
N = 200000
dt = 0.0005
n_steps = 15000
fwr = 6
# Desired particle density
baseStateVelocity = 5
# Normal restitution coefficient
alpha = 0.1
# System size and volume
LX = 200
LY = 200
LZ = 200
V = LX*LY*LZ
# Volume occuppied by one particle
particle_volume = (4/3)*np.pi*(effective_radius**3)
# La densidad se usa como volumen ocupado/V
# Particle density (packing fraction)
density = N*particle_volume/V

mean_free_path = 1 / (np.sqrt(2)*np.pi*(effective_diameter**2)*density)
knudsen_number = mean_free_path / min(LX, LY, LZ)
# Bins not used for the moment
n_bins = 50
bin_size = LZ/n_bins
ratio_bin_mfp = bin_size/mean_free_path


# Using Intel's Math Kernel Library
# https://software.intel.com/en-us/blogs/2016/06/15/faster-random-number-generation-in-intel-distribution-for-python
# 'MCG31' gives the best performance, 'MT2203' provides better randomness
# https://software.intel.com/en-us/mkl-vsnotes-basic-generators
random_intel.seed(brng='MT2203')

results = []
for alpha in (0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99):
#for alpha in (1,):
    n_runs = 12
    a2_mean = []
    for c in range(n_runs):
        # Initialize particle positions as a 2D numpy array (uniform).
        # As this is DSMC, some particles may be overlapping and not affect the outcome
        pos = random_intel.uniform(effective_radius, LX-effective_radius, (N, 3))
        # Initialize particle velocities as a 2D numpy array (normal/gaussian).
        vel = random_intel.normal(0, baseStateVelocity, (N, 3))
        # We now scale the velocity so that the vel of the center of mass 
        # is initialized at 0.  Pöschel pag.203
        vel -= np.mean(vel, axis=0)
       
        
        initial_mean_v = np.linalg.norm(vel, axis=1).mean()
        mean_free_time = mean_free_path / initial_mean_v     
        
        print('Number of particles: ', N)
        print('Coefficient of restitution: ', alpha)
        print('LX = ', LX)
        print('LY = ', LY)
        print('LZ = ', LZ)
        print('Density: ', density)
        print('Mean free path: ', mean_free_path)
        print('Knudsen number: ', knudsen_number)
        print('Mean free time: ', mean_free_time)
        print()
        
        temperatures = []
        cumulants = []
        n_collisions = 0
        rem = 0 # Remaining collisions (<1), see Poschel's Computational Granular Dynamics, pag 204 (dcollrest)
        for i in range(n_steps):
            v2 = np.linalg.norm(vel, axis=1)**2
            v2_mean = v2.mean()
            rv_max = findMaximumRelativeVelocity(v2_mean, fwr)
                        
            pos = propagate(dt, pos, vel, LX, LY, LZ)
            vel, cols_current_step, rem = computeCollisions(effective_diameter, 
                                                       alpha, V, N, rem, dt,
                                                       rv_max, pos, vel)
            n_collisions += cols_current_step
            cols_per_particle = n_collisions / N
            
            # Update a2 every 100 steps to keep permormance up
            if i%100==0:
                a2 = compute_a2(v2, 3)
                cumulants.append(a2)                
                T = (2/3)*v2_mean
                temperatures.append(T)

            printProgressBar(i, n_steps, prefix='Simulating system:', 
                             suffix='completed  -  T = '+str(T)+' - Run: '
                             +str(c+1)+' - CPP: '+str(cols_per_particle))
            
            # 40 collisions per particle should be more than enough
            if cols_per_particle >= 40:
                break

        # We average over the last quarter of the data
        a2_mean.append(np.mean(cumulants[-int(len(cumulants)/4):]))
        #plt.plot(cumulants)
        
    results.append([alpha, np.mean(a2_mean), np.std(a2_mean)])

print()    
#plt.plot(temperatures)
plt.plot(cumulants)
plotTheoryVsSimulation(results)
