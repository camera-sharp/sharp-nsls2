#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL sharpnsls2_ARRAY_API

#include "PySharpNSLS2.h"
#include <iostream>

PySharpNSLS2::PySharpNSLS2() {
}

// Recon API

void PySharpNSLS2::setAlpha(float v){
  SharpNSLS2::setAlpha(v);
}

void PySharpNSLS2::setBeta(float v){
  SharpNSLS2::setBeta(v);
}

void PySharpNSLS2::setStartUpdateProbe(int v){
  SharpNSLS2::setStartUpdateProbe(v);
}

void PySharpNSLS2::setStartUpdateObject(int v){
  SharpNSLS2::setStartUpdateObject(v);
}

void PySharpNSLS2::setAmpMax(float v){
  SharpNSLS2::setAmpMax(v); 
}

void PySharpNSLS2::setAmpMin(float v){
  SharpNSLS2::setAmpMin(v);   
}

void PySharpNSLS2::setPhaMax(float v){
  SharpNSLS2::setPhaMax(v);  
}

void PySharpNSLS2::setPhaMin(float v){
  SharpNSLS2::setPhaMin(v);  
}

// 


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

int PySharpNSLS2::setArgs(int argc, char * argv[]){
  
  int status = SharpNSLS2::setArgs(argc, argv);

  boost::multi_array<std::complex<float>, 2>& image = SharpNSLS2::getImage();

  int xdim = image.shape()[0];
  int ydim = image.shape()[1];

  m_image.resize(boost::extents[xdim][ydim]);
  return status;
}

int PySharpNSLS2::init(){
  return SharpNSLS2::init();
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





