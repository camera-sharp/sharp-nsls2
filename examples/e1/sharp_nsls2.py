#!/usr/bin/env python

# Example 1: The image and probe reconstruction from simulated frames

import sys
import time
from datetime import timedelta, datetime, tzinfo
import argparse
import numpy as np
import h5py
import sharp
import sharpnsls2
import matplotlib.pyplot as plt

sharpNSLS2 = sharpnsls2.PythonSharpNSLS2()

# Set the input parameters and initialize containers

niters = 101
args = ['local', '-o', '10',  '-i', '101', 'e1.cxi']

t1 = datetime.now();
sharpNSLS2.init(args)
t2 = datetime.now()

print ("initialization time: ", (t2 - t1))

# Run the reconstruction algorithm

t1 = datetime.now();
sharpNSLS2.run()
# for i in range(niters):
#    sharpNSLS2.step()   
t2 = datetime.now()

print ("reconstruction time: ", (t2 - t1))

# Write results of the reconstruction into the cxi file

sharpNSLS2.writeImage()

