#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


object = np.load("object.npy")
probe  = np.load("probe.npy")

fig = plt.figure()
plt.subplot(221)
plt.imshow(abs(object))
plt.subplot(222)
plt.imshow(np.angle(object))
plt.subplot(223)
plt.imshow(np.abs(probe))
plt.subplot(224)
plt.imshow(np.angle(probe))
plt.show()
# plt.savefig('e1.png')
