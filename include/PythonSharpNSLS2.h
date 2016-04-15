#pragma once

#ifndef PYTHON_SHARP_NSLS2_H
#define PYTHON_SHARP_NSLS2_H

#include "boost/multi_array.hpp"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <string>
#include <numpy/arrayobject.h>

#include "SharpNSLS2.h"

class PythonSharpNSLS2 : public SharpNSLS2 {

 public:

  PythonSharpNSLS2();

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
