# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 18:30:06 2018

@author: malopez
"""
import numpy as np
from numpy import random_intel

def computeCollisions(effective_diameter, effective_radius, alpha, V, N, rem, dt, rv_max, pos, vel):

    # First we have to determine the maximum number of candidate collisions
    # TODO. en un sitio divide por V (poschel) y en otro por 2V (garcia)
# =============================================================================
#     n_cols_max = ((N**2) * np.pi * (effective_diameter**2) * rv_max * dt / (2*V)) + rem
# =============================================================================
# =============================================================================
#     n_cols_max = ((N**2) * np.pi * (effective_diameter**2) * rv_max * dt / (V)) + rem
# =============================================================================
    n_cols_max = (N * rv_max * dt /2) + rem
    
    # Remaining collisions (<1) (to be computed in next time_step)
    rem = n_cols_max - int(n_cols_max)

    n_cols_max = int(n_cols_max)
    
    # It is more efficient to generate all random numbers at once
    random_intel.seed(brng='MT2203')
    # We choose multiple (n_cols_max) random pairs of particles
    random_pairs = random_intel.choice(N, size=(n_cols_max,2))
    # List of random numbers to use as collision criteria
    random_numbers = random_intel.uniform(0,1, n_cols_max)


    thetas = np.arccos(random_intel.uniform(-1,1, size=n_cols_max))
    phis = random_intel.uniform(0,2*np.pi, size=n_cols_max)
    
    x_coord = np.sin(thetas)*np.cos(phis)
    y_coord = np.sin(thetas)*np.sin(phis)
    z_coord = np.cos(thetas)
    sigmas = np.stack((x_coord, y_coord, z_coord), axis=1)
    
    
    # This is a vectorized method, it should be faster than the for loop
    # Testing, it is 2x faster
    # Using those random pairs we calculate relative velocities and its modulus
    rel_vs = np.array(list(map(lambda i, j: vel[i]-vel[j], random_pairs[:,0], random_pairs[:,1])))
    # TODO: ver notas de fvega, esta en la tabla
    # Scalar (dot) product of rel_vs and sigmas
    rel_vs_mod = np.array(list(map(lambda a, b: np.dot(a, b), rel_vs, sigmas)))
    
    # With this information we can check which collisions are valid
    ratios = rel_vs_mod / rv_max
    valid_cols = ratios > random_numbers
    
    # This is used to check that rv_max is always greater than the relative velocity of any given pair
    #alertas = np.where(ratios>1)[0]
    #if len(alertas) > 0:
    #    print(len(alertas))
    
    # The valid pairs of particles of each valid collision are:
    valid_pairs = random_pairs[valid_cols]
    
    # Number of collisions that take place in this step
    cols_current_step = len(valid_pairs)   

    # Valid directions (sigmas asociated with valid collisions)
    # We reverse the list to begin poping values from the beginning inside the for loop
    valid_sigmas = (sigmas[valid_cols])[::-1].tolist()

    # Now we only have to process those valid pairs
    for pair in valid_pairs:
        i, j = pair[0], pair[1]

        # We extract a random direction from the list that we generated previously
        sigma_ij = np.array(valid_sigmas.pop())
        # Alpha acts in the normal direction
        # Pöschel's 'Computational Granular Dynamics', pag 193:
        vel[i] -= 0.5*(1+alpha) * np.dot((vel[i] - vel[j]), sigma_ij) * sigma_ij
        vel[j] += 0.5*(1+alpha) * np.dot((vel[i] - vel[j]), sigma_ij) * sigma_ij

    return vel, cols_current_step, rem
    

def propagate(t, pos, vel, LX, LY, LZ):
    
    # Free stream of particles
    pos += t*vel
    
    # This is to account for the periodic boundaries
    pos[:,0] -= np.floor(pos[:,0]/LX)*LX
    pos[:,1] -= np.floor(pos[:,1]/LY)*LY
    pos[:,2] -= np.floor(pos[:,2]/LZ)*LZ
    
    return pos


def findMaximumRelativeVelocity(vel):
    """
    vmax = 0
    for i in range(n_bins):
        temp = np.where(pos[:,2] < (i+1)*bin_size)[0]
        particles = np.where(pos[temp,2] > i*bin_size)[0]
        
        local_vmax = v[particles].max()
        if vmax < local_vmax:
            vmax = local_vmax
    """
    v = np.linalg.norm(vel, axis=1)
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
