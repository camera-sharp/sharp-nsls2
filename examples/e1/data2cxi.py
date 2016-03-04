#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import h5py

# INPUT ##################################################

d = '../../data/d1/'

cxiFile = 'e1.cxi'

scanFile   = d + 'scan.npy'
objectFile = d + 'object.npy'
probeFile  = d + 'probe.npy'

energy     = 8000*1.60217657e-19 # 8 keV in J
pixel_size      =  55 * 1e-6 # 55 um pixels
real_pixel_size =  5.5 * 1e-9 # real space pixel size in m

# PROBE

probe = np.load(probeFile) # (384, 384)
shape = probe.shape
det_side = shape[1] # Detector side in pixels

# SCAN 

points = np.load(scanFile)
Y = points[0] # [-363, 362]
X = points[1] # [-363, 363]

X -= min(X) 
Y -= min(Y)

pixel_translation = np.column_stack((X, Y, np.zeros(Y.size)))
real_translation = pixel_translation * real_pixel_size

nframes = X.size

# OBJECT

object = np.load(objectFile) # (1170, 1172)
# plt.imshow(abs(object))
# plt.savefig('object.png')

# FRAMES

objFrames = np.empty((nframes,det_side,det_side), dtype = complex)
frames = np.empty((nframes,det_side,det_side))

for i in range(0,nframes):
    objFrames[i] = object[pixel_translation[i,1] : pixel_translation[i,1] + det_side, pixel_translation[i,0] : pixel_translation[i,0] + det_side]
    frames[i] = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(probe*objFrames[i]))))**2

# Calculated parameters

wavelength = 1.98644e-25/energy
distance   = (det_side * pixel_size * real_pixel_size)/ wavelength

corner_pos = [det_side/2*pixel_size,det_side/2*pixel_size,distance]

##########################################################

# OUTPUT

# Write out input file

f = h5py.File(cxiFile, "w")
f.create_dataset("cxi_version",data=140)
entry_1 = f.create_group("entry_1")

# 1. sample_1: name, geometry_1

sample_1   = entry_1.create_group("sample_1")

# 1.1 geometry_1: translation:

geometry_1 = sample_1.create_group("geometry_1")
geometry_1.create_dataset("translation", data=real_translation) # in meters

# 2. instrument_1: detector_1, source_1, data_1

instrument_1 = entry_1.create_group("instrument_1")

# 2.1 detector_1: distance, corner_position, x_pixel_size, y_pixel_size
# translation, data and axes, probe_mask
#'entry_1/instrument_1/detector_1/probe_mask'
detector_1 = instrument_1.create_group("detector_1")
detector_1.create_dataset("distance", data=distance) # in meters
detector_1.create_dataset("corner_position", data=corner_pos) # in meters
detector_1.create_dataset("x_pixel_size", data=pixel_size) # in meters
detector_1.create_dataset("y_pixel_size", data=pixel_size) # in meters

detector_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

data = detector_1.create_dataset("data",data=frames)
data.attrs['axes'] = "translation:y:x"

# 2.2 source_1: energy

source_1 = instrument_1.create_group("source_1")
source_1.create_dataset("energy", data=energy) # in J

source_1.create_dataset("probe",data=probe)

# 2.3 data_1: data, translationcd ..

data_1 = entry_1.create_group("data_1")

data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
data_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

f.close()

# Try to reconstruct with
# sharp-nsls2.bin  -o 1 -i 100  e1.cxi
# show.py
