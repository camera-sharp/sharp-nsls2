# SHARP-NSLS2

The NSLS-II specialization of the SHARP multi-GPU ptychographic program:

*  [SHARP: a distributed, GPU-based ptychographic solver](https://arxiv.org/abs/1602.01448), J. Appl. Cryst. 49, 2016
*  [Bringing HPC Reconstruction Algorithms to Big Data Platforms](http://ieeexplore.ieee.org/document/7747818/),
   NYSDS, New York, August 14-17, 2016

## Prerequisites

SHARP framework ([http://www.camera.lbl.gov/ptychography](http://www.camera.lbl.gov/ptychography)), including
associated prerequisites, such as cmake 3.7.0+, gcc 5.3.0+, boost 1.58+, fftw-3.3.5+, hdf5-1.8.17+, swig-3.0.10+, cuda 8.0+, mvapich2 2.2+.

## Installation

export FFTW3_HOME, SHARP_HOME

```
git clone git://github.com/camera-sharp/sharp-nsls2.git
mkdir build

cd build
cmake ../sharp-nsls2 -DCMAKE_INSTALL_PREFIX=<installation directory>
make
sudo make install

```

## Examples



