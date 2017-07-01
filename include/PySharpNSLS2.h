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

  PySharpNSLS2();

 public:

   void setGNode(); 

 public:

  int init(int argc, char * argv[]);

  int run();

  int step();

  PyObject* getImage();

  void writeImage();

  std::string getInputFile();

 protected:

  boost::multi_array<std::complex<float>, 2> m_image;

};

#endif
