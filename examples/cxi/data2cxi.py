#!/usr/bin/env python

# Example 1: The image and probe reconstruction from simulated frames

# This script (1) reads the scan, object and probe files;
# (2) generates frames (3) creates the cxi file for the Sharp-based applications

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import sharp

# Define the I/O parameters

inDir = '../../data/d1/'
outFile = 'e1.cxi'

# and the general parameters of the experiment

energyJ     = 8000*1.60217657e-19 # 8 keV in J
det_pixel_size  =  55 * 1e-6 # 55 um pixels
real_pixel_size =  5.5 * 1e-9 # real space pixel size in m

wavelength = 1.98644e-25/energyJ

# Load the probe file:

probeFile  = inDir + 'probe.npy'
probe = np.load(probeFile) # (384, 384)

# Define the size of the detector frame (assuming it is a square),
# calculate a distance to the detector, and define the position of
# the top left corner of the detector:

# Detector side in pixels
nx_prb = probe.shape[0] 
ny_prb = probe.shape[1] 

det_side = nx_prb

det_distance = (det_side * det_pixel_size * real_pixel_size)/ wavelength

# required by the input file, but not used
corner_pos = [det_side/2*det_pixel_size, det_side/2*det_pixel_size, det_distance]

# Define the initial probe:

init_prb_flag = False  #False to load a pre-existing array
init_prb = probe
init_prbFile = inDir + 'prb_init.npy'
init_prb = np.load(init_prbFile)

# Load a file with a sequence of the scan x-y points and shift them into
# the positive plane:

scanFile   = inDir + 'scan.npy'
points  = np.load(scanFile)

X = points[0] # [-363, 362]
Y = points[1] # [-363, 363]

X -= min(X) 
Y -= min(Y)

# transpose X and Y for the input file
# pixel_translation = np.column_stack((X, Y, np.zeros(Y.size)))
pixel_translation = np.column_stack((Y, X, np.zeros(Y.size)))
real_translation = pixel_translation * real_pixel_size

# Load the object file:

objectFile = inDir + 'object.npy'
object = np.load(objectFile) # (1170, 1172)

nx_obj = max(X) - min(X) + probe.shape[0] + 10
ny_obj = max(Y) - min(Y) + probe.shape[1] + 10

nx_obj = nx_obj + np.mod(nx_obj, 2)
ny_obj = ny_obj + np.mod(ny_obj, 2)

obj_min_x = 0
obj_max_x = int(max(X) - min(X)) + det_side

obj_min_y = object.shape[1] - (int(max(Y) - min(Y)) + det_side)
obj_max_y = object.shape[1]

# object = object[0: object.shape[0] - 61, 62: object.shape[1]]
object = object[obj_min_x: obj_max_x, obj_min_y: obj_max_y]

# Define the initial object:

nit_obj_flag = False  #False to load a pre-existing array
init_obj = object
init_objFile = inDir + 'obj_init.npy'
init_obj = np.load(init_objFile)

init_obj = init_obj[0: object.shape[0], 0: object.shape[1]]

# Now we can generate the detector frames from scan, object, and probe:

nframes = X.size

objFrames = np.empty((nframes,det_side,det_side), dtype = complex)
frames = np.empty((nframes,det_side,det_side))

for i in range(0,nframes):
    ix = int(X[i])
    iy = int(Y[i])
    objFrames[i] = object[ ix : ix + det_side, iy : iy + det_side]   
    frames[i] = np.abs(np.fft.fftshift(np.fft.fft2(probe*objFrames[i])))**2
    frames[i] = frames[i]/(det_side*det_side)

# Create the output file:

sharp.write_cxi(outFile, 
                energyJ, real_translation, det_distance, det_pixel_size, 
                frames, corner_position=corner_pos, 
                solution=None, illumination=init_prb, initial_image=init_obj,
                illumination_intensities_mask=None)

# Add the correction of the cxi file format

f = h5py.File(outFile, "r+")
source_1 = f['/entry_1/instrument_1/source_1']
source_1['probe'] = h5py.SoftLink('/entry_1/instrument_1/source_1/illumination')
f.close()





