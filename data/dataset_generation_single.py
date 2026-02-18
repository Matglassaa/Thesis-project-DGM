# file that uses flumy to generate datasets based on a prdefined parameter range
import os
import time
from flumy import *
import numpy as np
from flumy_utils import FlumyDataManager
import h5py

## Set grid parameters
nx = 250     #number of grid blockls in x direction
ny = 222    #number of grid blockls in y direction
mesh = 18   #size of grid blocks (in m) in both x and y direction
verbose = True

flsim = Flumy(nx, ny, mesh, verbose)

## Set hyperparameters for dataset generation
seed = 42
max_channel_depth = 5
net_gross = 70
max_sand_body_extention = 80
target_height = 10
vertical_resolution = 5     # in m
nz = int(target_height/vertical_resolution)
niter = -1

### Run simulation
print(f"Starting simulation to reach {target_height}m...")
flsim.launch(seed, hmax= max_channel_depth, isbx= max_sand_body_extention, ng= net_gross, zul= target_height, niter=niter)
print(f"simulation finished in ... seconds. \n")

#### Extract facies, grain size, and age data
fac,grain,age = flsim.getBlock(dz= vertical_resolution, zb=0 ,nz=nz)

##### Save to HDF5 Batch
batch_id = 1
n_samples = 1
output_dir = os.path.join("data","datasets","training")
manager = FlumyDataManager(output_dir=output_dir)

h5_path = manager.initialize_h5_file(batch_id, n_samples, nx, ny, nz)

# Save realization at index 0 of this batch file
manager.save_to_batch(h5_path, index=0, fac=fac, grain=grain, age=age)

# Verify integrity within the H5 container
manager.check_integrity(h5_path, index=0, original_data={'facies': fac, 'grain': grain, 'age': age})

print("Realization stored successfully in HDF5 format.")

# ##### Save generated dataset
# manager.save_realization(1, fac, grain, age)
# manager.check_integrity("realization_00001.npz", original_data={'facies': fac, 'grain': grain, 'age': age})