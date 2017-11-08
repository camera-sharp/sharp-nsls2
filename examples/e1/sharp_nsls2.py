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

sharpNSLS2 = sharpnsls2.PySharpNSLS2()

# Set the input parameters and initialize containers

niters = 101
args = ['local', '-o', '10',  '-i', '101', 'e1.cxi']

t1 = datetime.now();
sharpNSLS2.setArgs(args)
t2 = datetime.now()

print ("initialization time: ", (t2 - t1))

# Recon API: set the engine parameters

sharpNSLS2.setStartUpdateObject(0);
sharpNSLS2.setStartUpdateProbe(2);

sharpNSLS2.setBeta(0.9);

sharpNSLS2.setAmpMax(1.0);
sharpNSLS2.setAmpMin(0.0);
sharpNSLS2.setPhaMax(math.pi/2);
sharpNSLS2.setPhaMin(-math.pi/2);

# Run the reconstruction algorithm

sharpNSLS2.init()

t1 = datetime.now();
sharpNSLS2.run()
# for i in range(niters):
#    sharpNSLS2.step()   
t2 = datetime.now()

print ("reconstruction time: ", (t2 - t1))

# Write results of the reconstruction into the cxi file

sharpNSLS2.writeImage()

