#!/usr/bin/env python

# Example 1: The image and probe reconstruction from simulated frames

import sys
import time
from datetime import timedelta, datetime, tzinfo
import argparse
import math
import numpy as np
import h5py
import sharp
import sharpnsls2
import matplotlib.pyplot as plt

# Set the input parameters

niters = 51
inDir = '../../data/d1/'

z_m = 0.7495170470638932 # detector-to-sample distance (m)
ccd_pixel_um = 55 # detector pixel size (um)
lambda_nm = 0.15497979726416796 # x-ray wavelength (nm)

sharpNSLS2 = sharpnsls2.PySharpNSLS2()
sharpNSLS2.setZ(z_m)
sharpNSLS2.setLambda(lambda_nm)
sharpNSLS2.setPixelSize(ccd_pixel_um)

# Recon API: set Recon-specific parameters

sharpNSLS2.setStartUpdateObject(0);
sharpNSLS2.setStartUpdateProbe(2);

sharpNSLS2.setBeta(0.9);

sharpNSLS2.setAmpMax(1.0);
sharpNSLS2.setAmpMin(0.0);
sharpNSLS2.setPhaMax(math.pi/2);
sharpNSLS2.setPhaMin(-math.pi/2);

# sharpNSLS2.setChunks(4);

# Load and define an initial probe

init_prbFile = inDir + 'prb_init.npy'
init_prb = np.load(init_prbFile)

sharpNSLS2.setInitProbe(init_prb.astype(np.complex64))

print("initial probe: ", init_prb.shape)

# Load and define a scan

det_side = init_prb.shape[0]
real_pixel_size =  z_m*lambda_nm*1e-3/(det_side*ccd_pixel_um) # real space pixel size in m
corner_pos = [det_side/2*ccd_pixel_um*1e-6, det_side/2*ccd_pixel_um*1e-6, z_m]

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

# plt.scatter(X, Y)
# plt.show()

sharpNSLS2.setScan(real_translation.astype(np.float64))
print("scan: ", real_pixel_size, min(X), max(X), min(Y), max(Y))

# Load and define an initial object

init_objFile = inDir + 'obj_init.npy'
init_obj = np.load(init_objFile)

obj_max_x = int(max(X) - min(X)) + det_side
obj_max_y = int(max(Y) - min(Y)) + det_side

init_obj = init_obj[0: obj_max_x, 0: obj_max_y]
sharpNSLS2.setInitObject(init_obj.astype(np.complex64))

print("initial object: ", obj_max_x, obj_max_y)

# Generate and define frames

# Load a probe

prbFile = inDir + 'probe.npy'
probe = np.load(prbFile)

# plt.subplot(1,2,1)
# plt.imshow(abs(probe))
# plt.subplot(1,2,2)
# plt.imshow(np.angle(probe))

# Load an object

objectFile = inDir + 'object.npy'
object = np.load(objectFile) # (1170, 1172)

object = object[0: obj_max_x, 0: obj_max_y]

# plt.subplot(1,2,1)
# plt.imshow(abs(object))
# plt.subplot(1,2,2)
# plt.imshow(np.angle(object))

# Generate and define frames using scan, probe, and object

nframes = X.size

objFrames = np.empty((nframes,det_side,det_side), dtype = complex)
frames = np.empty((nframes,det_side,det_side))

for i in range(0,nframes):
    ix = int(X[i])
    iy = int(Y[i])
    objFrames[i] = object[ ix : ix + det_side, iy : iy + det_side]   
    frames[i] = np.abs(np.fft.fftshift(np.fft.fft2(probe*objFrames[i])))**2
    frames[i] = frames[i]/(det_side*det_side)
    
sharpNSLS2.setFrames(frames.astype(np.float32))

print("frames: ", nframes)

# Run the engine iterations

sharpNSLS2.init()

t1 = datetime.now();
for i in range(niters):
    sharpNSLS2.step()
    print(i, "obj_err: ", sharpNSLS2.getObjectError(),
          "prb_err: ", sharpNSLS2.getProbeError()) 
t2 = datetime.now()

print ("reconstruction time: ", (t2 - t1))

# Write results

object = sharpNSLS2.getObject()
np.save("object", object)

probe = sharpNSLS2.getProbe()
np.save("probe", probe)





