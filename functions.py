# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:30:06 2018

@author: malopez
"""
import math
import numpy as np
from numpy import random_intel

def computeCollisions(effective_diameter, effective_radius, alpha, V, N, Ne, dt, rv_max, pos, vel):
    # Define unity vectors:
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])
    # First we have to determine the maximum number of candidate collisions
    n_cols_max = int((N**2) * np.pi * (effective_diameter**2) * rv_max * Ne * dt / (2*V))
    
    # It is more efficient to generate all random numbers at once
    random_intel.seed(brng='MT2203')
    # We choose multiple (n_cols_max) random pairs of particles
    random_pairs = random_intel.choice(N, size=(n_cols_max,2))
    # List of random numbers to use as collision criteria
    random_numbers = random_intel.uniform(0,1, n_cols_max).tolist()
    # We iterate over the list of possible collisions (over each pair of particles)
    for pair in random_pairs:
        # We choose a random pair of particles
        i, j = pair[0], pair[1]
        # Computing their relative velocity
        rv = relativeVelocity(i, j, vel)
        # We extract a random number from the list
        rand = random_numbers.pop()
        # We decide whether the collision takes place or not
        if rv/rv_max > rand:
            # Proccess that collision:

            v_center_mass = 0.5 * (vel[i] + vel[j])
            q = random_intel.uniform(-1,1)
            theta = np.arccos(q)
            phi = random_intel.uniform(0, 2*np.pi)
            v_prime = rv * (np.sin(theta)*np.cos(phi)*x + np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z)
            # 3D velocity calculation, taken from 'Alexander and Garcia, Computer Simulation, DSMC'
            # TODO: Lo del alpha esta mal, solo se aplica a la componente normal
            vel[i] = alpha*(v_center_mass + 0.5*v_prime)
            vel[j] = alpha*(v_center_mass - 0.5*v_prime)
    
    return vel
    

def propagate(t, pos, vel, LX, LY, LZ):
    
    # Free stream of particles
    pos += t*vel
    
    # This is to account for the periodic boundaries
    pos[:,0] -= np.floor(pos[:,0]/LX)*LX
    pos[:,1] -= np.floor(pos[:,1]/LY)*LY
    pos[:,2] -= np.floor(pos[:,2]/LZ)*LZ
    
    return pos


def findMaximumRelativeVelocity(v):
    """
    vmax = 0
    for i in range(n_bins):
        temp = np.where(pos[:,2] < (i+1)*bin_size)[0]
        particles = np.where(pos[temp,2] > i*bin_size)[0]
        
        local_vmax = v[particles].max()
        if vmax < local_vmax:
            vmax = local_vmax
    """
    vmax = v.max()    
    return 2*vmax


def relativeVelocity(i, j, vel):
    
    rel_v = vel[i] - vel[j]
    modulus = np.sqrt(rel_v[0]**2 + rel_v[1]**2 + rel_v[2]**2)
    
    return modulus


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def computeKurtosis(vel):
    v2_sep = vel*vel
    v2 = v2_sep[:,0] + v2_sep[:,1]
    # v can also come in handy for calculating the kurtosis later-on
    v = np.vectorize(math.sqrt)(v2)
    k = (v**4).mean()/((v**2).mean())**2
    
    return k

def compute_a2(vel, dimensions):
    kurtosis = computeKurtosis(vel)
    a2 = (dimensions/(dimensions+2))*kurtosis -1
    
    return a2
