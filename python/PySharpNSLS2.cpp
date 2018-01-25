#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL sharpnsls2_ARRAY_API

#include "PySharpNSLS2.h"
#include <iostream>

PySharpNSLS2::PySharpNSLS2() {
}

// Recon Input API

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

void PySharpNSLS2::setZ(float v){
  SharpNSLS2::setZ(v);  
}

void PySharpNSLS2::setLambda(float v){
  SharpNSLS2::setLambda(v);  
}

void PySharpNSLS2::setPixelSize(float v){
  SharpNSLS2::setPixelSize(v);  
}


//

void PySharpNSLS2::setScan(PyObject* pyObject){

  PyArrayObject* pyArray = (PyArrayObject*) pyObject;
  
  int ndims = PyArray_NDIM(pyArray);
  npy_intp* shape = PyArray_DIMS(pyArray);
  int xdim = shape[0];
  int ydim = shape[1];
  double* data = (double*) PyArray_DATA(pyArray);
  
  boost::multi_array<double, 2> scan;
  scan.resize(boost::extents[xdim][ydim]);
  double* dest = scan.data();
  
  for(int i = 0; i < xdim*ydim; i++){
    dest[i] = data[i];
  }
  
  SharpNSLS2::setScan(scan);  
}

void PySharpNSLS2::setFrames(PyObject* pyFrames){

  PyArrayObject* pyArray = (PyArrayObject*) pyFrames;
  
  int ndims = PyArray_NDIM(pyArray);
  npy_intp* shape = PyArray_DIMS(pyArray);
  int nframes = shape[0];
  int xdim    = shape[1];
  int ydim    = shape[2];
  float* data = (float*) PyArray_DATA(pyArray);

  boost::multi_array<float, 3> frames;
  frames.resize(boost::extents[nframes][xdim][ydim]);
  float* dest = frames.data();
  
  for(int i = 0; i < nframes*xdim*ydim; i++){
    dest[i] = data[i];
  }

  SharpNSLS2::setFrames(frames);  
}

void PySharpNSLS2::setInitObject(PyObject* pyObject){

  PyArrayObject* pyArray = (PyArrayObject*) pyObject;
  
  int ndims = PyArray_NDIM(pyArray);
  npy_intp* shape = PyArray_DIMS(pyArray);
  int xdim = shape[0];
  int ydim = shape[1];
  std::complex<float>* data = (std::complex<float>*) PyArray_DATA(pyArray);
  
  boost::multi_array<std::complex<float>, 2> initObject;
  initObject.resize(boost::extents[xdim][ydim]);
  std::complex<float>* dest = initObject.data();
  
  for(int i = 0; i < xdim*ydim; i++){
    dest[i] = data[i];
  }
  
  SharpNSLS2::setInitObject(initObject);  
  
}

void PySharpNSLS2::setInitProbe(PyObject* pyObject){
  PyArrayObject* pyArray = (PyArrayObject*) pyObject;
  
  int ndims = PyArray_NDIM(pyArray);
  npy_intp* shape = PyArray_DIMS(pyArray);
  int xdim = shape[0];
  int ydim = shape[1];

  // PyArray_Descr* descr = PyArray_DTYPE(PyArrayObject* arr);
  // std::cout << "setInitProbe: ndims: " << ndims << ", xdim: " << xdim << ", ydim: " << ydim << std::endl;
  
  std::complex<float>* data = (std::complex<float>*) PyArray_DATA(pyArray);
  
  boost::multi_array<std::complex<float>, 2> initProbe;
  initProbe.resize(boost::extents[xdim][ydim]);
  std::complex<float>* dest = initProbe.data();
  
  for(int i = 0; i < xdim*ydim; i++){
    dest[i] = data[i];
  }
  
  SharpNSLS2::setInitProbe(initProbe);  
}

// Recon Output API

PyObject* PySharpNSLS2::getObject(){

  boost::multi_array<std::complex<float>, 2>& image = SharpNSLS2::getObject();

  int xdim = image.shape()[0];
  int ydim = image.shape()[1];

  // std::cout << "object: xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_object.resize(boost::extents[xdim][ydim]);
  m_object = image;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_object.data()); 
}

PyObject* PySharpNSLS2::getProbe(){

  boost::multi_array<std::complex<float>, 2>& image = SharpNSLS2::getProbe();

  int xdim = image.shape()[0];
  int ydim = image.shape()[1];

  // std::cout << "probe: xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_probe.resize(boost::extents[xdim][ydim]);
  m_probe = image;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_probe.data()); 
}

float PySharpNSLS2::getObjectError(){
  return SharpNSLS2::getObjectError();  
}

float PySharpNSLS2::getProbeError(){
  return SharpNSLS2::getProbeError();  
}

// SHARP internal containers

PyObject* PySharpNSLS2::getProducts(){

  boost::multi_array<std::complex<float>, 3>& products = SharpNSLS2::getFrames();

  int n    = products.shape()[0];
  int xdim = products.shape()[1];
  int ydim = products.shape()[2];

  std::cout << "products: n: " << n << ", xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_products.resize(boost::extents[n][xdim][ydim]);
  m_products = products;

  npy_intp dims[] = {n, xdim, ydim};
  return PyArray_SimpleNewFromData(3, dims, NPY_COMPLEX64, (void *) m_products.data()); 
}

PyObject* PySharpNSLS2::getFramesCorners(){

  boost::multi_array<std::complex<float>, 1>& corners = SharpNSLS2::getFramesCorners();

  int n    = corners.shape()[0];
 
  std::cout << "corners: n: " << n << std::endl;

  m_corners.resize(boost::extents[n]);
  m_corners = corners;

  npy_intp dims[] = {n};
  return PyArray_SimpleNewFromData(1, dims, NPY_COMPLEX64, (void *) m_corners.data()); 
}

PyObject* PySharpNSLS2::getOverlapingFrames(){

  boost::multi_array<int, 1>& overlaping_frames = SharpNSLS2::getOverlapingFrames();

  int n    = overlaping_frames.shape()[0];
 
  std::cout << "overlaping_frames: n: " << n << std::endl;

  m_overlaping_frames.resize(boost::extents[n]);
  m_overlaping_frames = overlaping_frames;

  npy_intp dims[] = {n};
  return PyArray_SimpleNewFromData(1, dims, NPY_INT, (void *) m_overlaping_frames.data()); 
}

PyObject* PySharpNSLS2::getOverlapingFramesIndex(){

  boost::multi_array<int, 1>& overlaping_frames_index = SharpNSLS2::getOverlapingFramesIndex();

  int n    = overlaping_frames_index.shape()[0];
 
  std::cout << "overlaping_frames_index: n: " << n << std::endl;

  m_overlaping_frames_index.resize(boost::extents[n]);
  m_overlaping_frames_index = overlaping_frames_index;

  npy_intp dims[] = {n};
  return PyArray_SimpleNewFromData(1, dims, NPY_INT, (void *) m_overlaping_frames_index.data()); 
}

PyObject* PySharpNSLS2::getImageScale(){

  boost::multi_array<std::complex<float>, 2>& imageScale = SharpNSLS2::getImageScale();

  int xdim = imageScale.shape()[0];
  int ydim = imageScale.shape()[1];

  // std::cout << "object: xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_imageScale.resize(boost::extents[xdim][ydim]);
  m_imageScale = imageScale;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_imageScale.data()); 
}

PyObject* PySharpNSLS2::getIlluminatedArea(){

  boost::multi_array<std::complex<float>, 2>& area = SharpNSLS2::getIlluminatedArea();

  int xdim = area.shape()[0];
  int ydim = area.shape()[1];

  // std::cout << "object: xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_illuminatedArea.resize(boost::extents[xdim][ydim]);
  m_illuminatedArea = area;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_illuminatedArea.data()); 
}

PyObject* PySharpNSLS2::getIlluminationNumerator(){

  boost::multi_array<std::complex<float>, 2>& numerator = SharpNSLS2::getIlluminationNumerator();

  int xdim = numerator.shape()[0];
  int ydim = numerator.shape()[1];

  // std::cout << "object: xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_illuminationNumerator.resize(boost::extents[xdim][ydim]);
  m_illuminationNumerator = numerator;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_illuminationNumerator.data()); 
}

PyObject* PySharpNSLS2::getIlluminationDenominator(){

  boost::multi_array<std::complex<float>, 2>& denominator = SharpNSLS2::getIlluminationDenominator();

  int xdim = denominator.shape()[0];
  int ydim = denominator.shape()[1];

  // std::cout << "object: xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_illuminationDenominator.resize(boost::extents[xdim][ydim]);
  m_illuminationDenominator = denominator;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_illuminationDenominator.data()); 
}

PyObject* PySharpNSLS2::getPrbObj(){

  boost::multi_array<std::complex<float>, 3>& prb_obj = SharpNSLS2::getPrbObj();

  int n    = prb_obj.shape()[0];
  int xdim = prb_obj.shape()[1];
  int ydim = prb_obj.shape()[2];

  std::cout << "prb_obj: n: " << n << ", xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_prb_obj.resize(boost::extents[n][xdim][ydim]);
  m_prb_obj = prb_obj;

  npy_intp dims[] = {n, xdim, ydim};
  return PyArray_SimpleNewFromData(3, dims, NPY_COMPLEX64, (void *) m_prb_obj.data()); 
}

PyObject* PySharpNSLS2::getTmp2(){

  boost::multi_array<std::complex<float>, 3>& tmp2 = SharpNSLS2::getTmp2();

  int n    = tmp2.shape()[0];
  int xdim = tmp2.shape()[1];
  int ydim = tmp2.shape()[2];

  std::cout << "tmp2: n: " << n << ", xdim: " << xdim << ", ydim: " << ydim << std::endl;

  m_tmp2.resize(boost::extents[n][xdim][ydim]);
  m_tmp2 = tmp2;

  npy_intp dims[] = {n, xdim, ydim};
  return PyArray_SimpleNewFromData(3, dims, NPY_COMPLEX64, (void *) m_tmp2.data()); 
}


// MPI/GPU interface

int PySharpNSLS2::getRank(){
  return SharpNSLS2::getRank();  
}

void PySharpNSLS2::setGNode(){
  return SharpNSLS2::setGNode();
}

void PySharpNSLS2::setChunks(int v){
  SharpNSLS2::setChunks(v);  
}

//

int PySharpNSLS2::setArgs(int argc, char * argv[]){
  
  int status = SharpNSLS2::setArgs(argc, argv);
  return status;
}

int PySharpNSLS2::init(){
  return SharpNSLS2::init();
}

int PySharpNSLS2::step(){
  return SharpNSLS2::step();
}

// Depricated interface


int PySharpNSLS2::run(){
  return SharpNSLS2::run();
}

PyObject* PySharpNSLS2::getImage(){

  boost::multi_array<std::complex<float>, 2>& image = SharpNSLS2::getImage();

  int xdim = image.shape()[0];
  int ydim = image.shape()[1];

  m_image.resize(boost::extents[xdim][ydim]);
  m_image = image;

  npy_intp dims[] = {xdim, ydim};
  return PyArray_SimpleNewFromData(2, dims, NPY_COMPLEX64, (void *) m_image.data()); 
}

void PySharpNSLS2::writeImage(){
  return SharpNSLS2::writeImage();
}

std::string PySharpNSLS2::getInputFile(){
  return SharpNSLS2::getInputFile();
}





