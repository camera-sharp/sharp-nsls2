#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import h5py

f = h5py.File('e1.cxi','r')
fig = plt.figure()
plt.subplot(221)
plt.imshow(abs(np.array(f['/entry_1/image_1/data'])))
plt.subplot(222)
plt.imshow(np.angle(np.array(f['/entry_1/image_1/data'])))
plt.subplot(223)
plt.imshow(np.abs(np.array(f['/entry_1/image_1/process_1/final_illumination'])))
plt.subplot(224)
plt.imshow(np.angle(np.array(f['/entry_1/image_1/process_1/final_illumination'])))
plt.show()
# plt.savefig('e1.png')
