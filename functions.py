# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:30:06 2018

@author: malopez
"""
import numpy as np
from numpy import random_intel

def computeCollisions(effective_diameter, effective_radius, alpha, V, N, Ne, dt, rv_max, pos, vel):
    """
    # Define unity vectors:
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])
    """
    # First we have to determine the maximum number of candidate collisions
    n_cols_max = int((N**2) * np.pi * (effective_diameter**2) * rv_max * Ne * dt / (2*V))
    # Another method of estimating that value, a numerical factor times the average thermal velocity
    
    
    
    # It is more efficient to generate all random numbers at once
    random_intel.seed(brng='MT2203')
    # We choose multiple (n_cols_max) random pairs of particles
    random_pairs = random_intel.choice(N, size=(n_cols_max,2))
    # List of random numbers to use as collision criteria
    random_numbers = random_intel.uniform(0,1, n_cols_max).tolist()
    
    
    #"""
    # This is a vectorized method, it should be faster than the for loop
    # Testing, it is 2x faster
    # Using those random pairs we calculate relative velocities and its modulus
    rel_vs = np.array(list(map(lambda i, j: vel[i]-vel[j], random_pairs[:,0], random_pairs[:,1])))
    rel_vs_mod = np.linalg.norm(rel_vs, axis=1)
    # With this information we can check which collisions are valid
    ratios = rel_vs_mod / rv_max
    valid_cols = ratios > random_numbers
    
    # This is used to check that rv_max is always greater than the relative velocity of any given pair
    #alertas = np.where(ratios>1)[0]
    #if len(alertas) > 0:
    #    print(len(alertas))
    
    valid_pairs = random_pairs[valid_cols]
    # Number of collisions that take place in this step
    cols_current_step = len(valid_pairs)
    # Now we only have to process those valid pairs
    for pair in valid_pairs:
        i, j = pair[0], pair[1]
        """
        # The normal direction shall be calculated using random numbers
        q = random_intel.uniform(-1,1)
        theta = np.arccos(q)
        phi = random_intel.uniform(0, 2*np.pi)
        sigma_ij = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        sigma_ij = sigma_ij / np.linalg.norm(sigma_ij)
        """
        # The normal direction shall be calculated using random numbers
        # TODO: this numbers and vectors can be generated more efficiently ouside the loop
        # https://math.stackexchange.com/questions/1385137/calculate-3d-vector-out-of-two-angles-and-vector-length
        theta = random_intel.uniform(0, 2*np.pi)
        phi = random_intel.uniform(0, 2*np.pi)
        sigma_ij = np.array([np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi), np.sin(theta)])
        # Next line is redundant, the vector is already unitary
        # sigma_ij = sigma_ij / np.linalg.norm(sigma_ij)
        
        """
        # First, we calculate the normal direction as a unit vector
        sigma_ij = pos[i] - pos[j]
        sigma_ij = sigma_ij / np.linalg.norm(sigma_ij)
        """
        # Alpha acts in the normal direction
        # Pöschel's 'Computational Granular Dynamics', pag 193:
        vel[i] -= 0.5*(1+alpha) * np.dot((vel[i] - vel[j]), sigma_ij) * sigma_ij
        vel[j] += 0.5*(1+alpha) * np.dot((vel[i] - vel[j]), sigma_ij) * sigma_ij
    #"""
    """
    # This is another slower method, it is here for future reference
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

            # 3D velocity calculation, taken from 'Alexander and Garcia, Computer Simulation, DSMC'
            v_center_mass = 0.5 * (vel[i] + vel[j])
            q = random_intel.uniform(-1,1)
            theta = np.arccos(q)
            phi = random_intel.uniform(0, 2*np.pi)
            v_prime = rv * (np.sin(theta)*np.cos(phi)*x + np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z)
            
            # TODO: There is an error, alpha should only act in the normal direction
            vel[i] = v_center_mass + 0.5*v_prime
            vel[j] = v_center_mass - 0.5*v_prime
            
            
    """
    return vel, cols_current_step
    

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
    # I overestimate, returning 2 times the maximum velocity in the system
    return 2*vmax


def relativeVelocity(i, j, vel):
    
    rel_v = vel[i] - vel[j]
    modulus = np.linalg.norm(rel_v)
    
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
    v = np.linalg.norm(vel, axis=1)
    v2 = v**2

    k = (v**4).mean()/(v2.mean())**2
    
    return k

def compute_a2(vel, dimensions):
    kurtosis = computeKurtosis(vel)
    a2 = (dimensions/(dimensions+2))*kurtosis - 1
    
    return a2


def theoretical_a2(alpha, d, method=2):
    if method==1:
        a2 = (16*(1-alpha) * (1 - 2*(alpha**2))) / (9 + 24*d - alpha*(41 - 8*d) + 30*(1-alpha)*(alpha**2))
    elif method==2:
        a2 = (16*(1-alpha) * (1 - 2*(alpha**2))) / (25 + 24*d - alpha*(57 - 8*d) - 2*(1-alpha)*(alpha**2))
        
    
    return a2
