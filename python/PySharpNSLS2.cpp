#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL sharpnsls2_ARRAY_API

#include "PySharpNSLS2.h"
#include <iostream>

PySharpNSLS2::PySharpNSLS2() {
}


PyObject* PySharpNSLS2::getImage(){

  boost::multi_array<std::complex<float>, 2>& image = SharpNSLS2::getImage();

  int xdim = image.shape()[0];
  int ydim = image.shape()[1];

  m_image = image;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_image.data()); 
}

void PySharpNSLS2::setGNode(){
  return SharpNSLS2::setGNode();
}

int PySharpNSLS2::init(int argc, char * argv[]){
  int status = SharpNSLS2::init(argc, argv);

  boost::multi_array<std::complex<float>, 2>& image = SharpNSLS2::getImage();

  int xdim = image.shape()[0];
  int ydim = image.shape()[1];

  m_image.resize(boost::extents[xdim][ydim]);
  return status;
}


int PySharpNSLS2::run(){
  return SharpNSLS2::run();
}

int PySharpNSLS2::step(){
  return SharpNSLS2::step();
}

void PySharpNSLS2::writeImage(){
  return SharpNSLS2::writeImage();
}

std::string PySharpNSLS2::getInputFile(){
  return SharpNSLS2::getInputFile();
}





