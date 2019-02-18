import numpy as np

# =========Simulation Variables=========

m = 1 # Mass
N = 100000 # Total number of particles in the system
dt = 0.001 # Time step duration
n_steps = 15000 # Number of steps
n_runs = 20 # Number of runs for each alpha
fwr = 6 # Parameter used for calculating maximum relative velocity
baseStateVelocity = 5 # Used to initialize velocity
alpha = 0.1 # Normal restitution coefficient

# System size and volume
LX = 200
LY = 200
LZ = 200
V = LX*LY*LZ

effective_diameter = 1
effective_radius = effective_diameter/2

particle_volume = (4/3)*np.pi*(effective_radius**3) # Volume occuppied by one particle
density = N*particle_volume/V # Particle density (packing fraction)

mean_free_path = 1 / (np.sqrt(2)*np.pi*(effective_diameter**2)*density)
knudsen_number = mean_free_path / min(LX, LY, LZ)

# Bins, not used for the moment
n_bins = 50
bin_size = LZ/n_bins
ratio_bin_mfp = bin_size/mean_free_path