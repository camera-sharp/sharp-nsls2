#pragma once

#ifndef PY_SHARP_NSLS2_H
#define PY_SHARP_NSLS2_H

#include "boost/multi_array.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <string>
#include <numpy/arrayobject.h>

#include "SharpNSLS2.h"

class PySharpNSLS2 : public SharpNSLS2 {

 public:

  /** Constructor */
  PySharpNSLS2();

 public:

  // MPI/GPU interface

  int getRank();

  void setGNode();

  void setChunks(int v);

 public:

  /** set the SHARP variables with command arguments */
  int setArgs(int argc, char * argv[]);

 public:

  // Recon Input API

  /** general feedback parameter */
  void setBeta(float v);

  /** espresso threshold coefficient */
  void setAlpha(float v);

  /** iteration number start updating probe */
  void setStartUpdateProbe(int v);

  /** iteration number start updating object */
  void setStartUpdateObject(int v);

  /** maximum object magnitude */
  void setAmpMax(float v);

  /** minimum object magnitude */
  void setAmpMin(float v);

  /** maximum object phase */
  void setPhaMax(float v);

  /** minimum object phase */
  void setPhaMin(float v);

  /** detector distance [m] */
  void setZ(float v);

  /** wavelength, nm */
  void setLambda(float v);

  /** ccd pixel size, um */
  void setPixelSize(float v);

 public:

  /** scan points (2d array of doubles) */
  void setScan(PyObject* object);

  /** detector frames (3d array of floats) */
  void setFrames(PyObject* object);

  /** initial object (2d array of complex floats) */
  void setInitObject(PyObject* object);

  /** initial probe (2d array of complex floats) */
  void setInitProbe(PyObject* probe);

 public:

    // Recon Output API

    PyObject* getObject();

    PyObject* getProbe();

    float getObjectError();

    float getProbeError();

    // Sharp

    PyObject* getProducts();

    PyObject* getFramesCorners();

    PyObject* getImageScale();

    PyObject* getIlluminatedArea();

    PyObject* getOverlapingFrames();

    PyObject* getOverlapingFramesIndex();

    PyObject* getIlluminationNumerator();

    PyObject* getIlluminationDenominator();

    PyObject* getPrbObj();

    PyObject* getTmp2();    

 public:

  /** Initialize engine */
  int init();

  /** Run one iteration */
  int step();

 public:

  // Depricated interface

  /** Run all iterations */
  int run();  

  PyObject* getImage();

  void writeImage();

  std::string getInputFile();

 protected:

  boost::multi_array<std::complex<float>, 2> m_image;
  boost::multi_array<std::complex<float>, 2> m_object;
  boost::multi_array<std::complex<float>, 2> m_probe;

  boost::multi_array<std::complex<float>, 1> m_corners;
  boost::multi_array<std::complex<float>, 3> m_products;

  boost::multi_array<int, 1> m_overlaping_frames;
  boost::multi_array<int, 1> m_overlaping_frames_index;
  boost::multi_array<std::complex<float>, 2> m_imageScale;
  boost::multi_array<std::complex<float>, 2> m_illuminatedArea;

  boost::multi_array<std::complex<float>, 2> m_illuminationNumerator;
  boost::multi_array<std::complex<float>, 2> m_illuminationDenominator;

  boost::multi_array<std::complex<float>, 3> m_prb_obj;
  boost::multi_array<std::complex<float>, 3> m_tmp2; 
  
  
};

#endif
