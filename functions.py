# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:30:06 2018

@author: malopez
"""
import numpy as np
from numpy import random_intel

def computeCollisions(effective_diameter, alpha, V, N, rem, dt, rv_max, pos, vel):

    # First we have to determine the maximum number of candidate collisions   
    n_cols_max = (N * rv_max * dt /2) + rem
    
    # Remaining collisions (<1) (to be computed in next time_step)
    rem = n_cols_max - int(n_cols_max)
    # We only use the integer part
    n_cols_max = int(n_cols_max)
    
    # It is more efficient to generate all random numbers at once
    random_intel.seed(brng='MT2203')
    # We choose multiple (n_cols_max) random pairs of particles
    random_pairs = random_intel.choice(N, size=(n_cols_max,2))
    # List of random numbers to use as collision criteria
    random_numbers = random_intel.uniform(0,1, n_cols_max)

    # Now, we generate random directions, (modulus 1) sigmas
    costheta = random_intel.uniform(0,2, size=n_cols_max) - 1
    sintheta = np.sqrt(1-costheta**2)
    phis = random_intel.uniform(0,2*np.pi, size=n_cols_max)
    
    x_coord = sintheta*np.cos(phis)
    y_coord = sintheta*np.sin(phis)
    z_coord = costheta
    sigmas = np.stack((x_coord, y_coord, z_coord), axis=1)   
    
    # This is a vectorized method, it should be faster than the for loop
    # Using those random pairs we calculate relative velocities
    rel_vs = np.array(list(map(lambda i, j: vel[i]-vel[j], random_pairs[:,0], random_pairs[:,1])))
    # And now its modulus by performing a dot product with sigmas array
    rel_vs_mod = np.sum(rel_vs*sigmas, axis=1)
    
    # With this information we can check which collisions are valid
    ratios = rel_vs_mod / rv_max
    valid_cols = ratios > random_numbers
    
    # The valid pairs of particles of each valid collision are:
    valid_pairs = random_pairs[valid_cols]
    
    # Number of collisions that take place in this step
    cols_current_step = len(valid_pairs)   

    # Now, we select only those rows that correspond to valid collisions
    valid_dotProducts = rel_vs_mod[valid_cols]
    # See: https://stackoverflow.com/questions/16229823/how-to-multiply-numpy-2d-array-with-numpy-1d-array
    valid_vectors = sigmas[valid_cols] * valid_dotProducts[:, None]
    new_vel_components = 0.5*(1+alpha) * valid_vectors
    
    valid_is = valid_pairs[:,0]
    valid_js = valid_pairs[:,1]
    
    # Updating the velocities array with its new values
    vel[valid_is] -= new_vel_components
    vel[valid_js] += new_vel_components

    return vel, cols_current_step, rem
    

def propagate(t, pos, vel, LX, LY, LZ):
    
    # Free stream of particles
    pos += t*vel
    
    # This is to account for the periodic boundaries
    pos[:,0] -= np.floor(pos[:,0]/LX)*LX
    pos[:,1] -= np.floor(pos[:,1]/LY)*LY
    pos[:,2] -= np.floor(pos[:,2]/LZ)*LZ
    
    return pos


def findMaximumRelativeVelocity(v2_mean, fwr):
    
    vmax = fwr*np.sqrt(2*v2_mean/3)

    return vmax


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
    
def computeKurtosis(v2):
    v4 = v2**2

    k = (v4).mean()/(v2.mean())**2
    
    return k

def compute_a2(v2, dimensions):
    kurtosis = computeKurtosis(v2)
    a2 = (dimensions/(dimensions+2))*kurtosis - 1
    
    return a2


def theoretical_a2(alpha, d, method=2):
    if method==1:
        a2 = (16*(1-alpha) * (1 - 2*(alpha**2))) / (9 + 24*d - alpha*(41 - 8*d) + 30*(1-alpha)*(alpha**2))
    elif method==2:
        a2 = (16*(1-alpha) * (1 - 2*(alpha**2))) / (25 + 24*d - alpha*(57 - 8*d) - 2*(1-alpha)*(alpha**2))
        
    
    return a2
