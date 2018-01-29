#include <boost/assert.hpp>
#include <input_output.h>
#include <communicator_mpi.h>
#include <strategy.h>
#include <CudaEngineDM.h>
#include <solver.h>
#include <string>
#include <unistd.h>
#include <options.h>
#include <exception>
#include "counter.h"

#include "range.h"

#include "SharpNSLS2.h"
#include "CommunicatorGNode.h"

SharpNSLS2::SharpNSLS2()
  : m_center(2,0), m_reciprocal_size(2,0) {

  // Recon input parameters

   m_start_update_object = 0;
   m_start_update_probe  = 2;

   m_alpha = 1.e-8; // espresso threshold coefficient
   m_beta  = 0.9;   // general feedback parameter

   m_amp_max =  1.0;
   m_amp_min =  0.0;
   m_pha_max =  3.14/2;
   m_pha_min = -3.14/2;

   m_has_scan = false;
   m_has_frames = false;
   m_has_init_probe = false;
   m_has_init_object = false;

   //

   m_z_m = 0.0;
   m_lambda_nm = 0.0;
   m_ccd_pixel_um = 0.0;

   // MPI/GPU

  isGNode = false;
  
  m_chunks = 1;

  // 

  m_engine = 0;
  m_communicator = 0;
  m_input_output = 0;
  m_strategy = 0;
  // m_solver = 0;

}

SharpNSLS2::~SharpNSLS2() {
  clean();
  if(m_communicator) delete m_communicator;
}

// Recon API

void SharpNSLS2::setAlpha(float v){
     m_alpha = v;
}

void SharpNSLS2::setBeta(float v){
     m_beta = v;
}

void SharpNSLS2::setStartUpdateProbe(int v){
     m_start_update_probe = v;
}

void SharpNSLS2::setStartUpdateObject(int v){
     m_start_update_object = v;
}

void SharpNSLS2::setAmpMax(float v){
     m_amp_max = v;
}

void SharpNSLS2::setAmpMin(float v){
     m_amp_min = v;
}

void SharpNSLS2::setPhaMax(float v){
     m_pha_max = v;
}

void SharpNSLS2::setPhaMin(float v){
     m_pha_min = v;
}

void SharpNSLS2::setZ(float v){
     m_z_m = v;
}

void SharpNSLS2::setLambda(float v){
     m_lambda_nm = v;
}

void SharpNSLS2::setPixelSize(float v){
     m_ccd_pixel_um = v;
}

//

void SharpNSLS2::setScan(const boost::multi_array<double, 2> & scan){
     m_has_scan = true;
     int xdim = scan.shape()[0];
     int ydim = scan.shape()[1];
     m_all_translations.resize(boost::extents[xdim][ydim]);
     m_all_translations = scan;
}

void SharpNSLS2::setFrames(const boost::multi_array<float, 3>& frames){

    m_has_frames = true;

   // see ThrustEngine::setFrames()
   
   m_total_n_frames = frames.shape()[0];
   m_frames_height  = frames.shape()[1];
   m_frames_width   = frames.shape()[2];
 
   m_frames.resize(boost::extents[m_total_n_frames][m_frames_height][m_frames_width]);
   m_frames = frames;
}

void SharpNSLS2::setInitProbe(const boost::multi_array<std::complex<float>, 2> & probe){
     m_has_init_probe = true;
     int xdim = probe.shape()[0];
     int ydim = probe.shape()[1];
     m_init_probe.resize(boost::extents[xdim][ydim]);
     m_init_probe = probe;
}

void SharpNSLS2::setInitObject(const boost::multi_array<std::complex<float>, 2> & object){
     m_has_init_object = true;
     int xdim = object.shape()[0];
     int ydim = object.shape()[1];
     m_init_object.resize(boost::extents[xdim][ydim]);
     m_init_object = object;
}

// MPI/GPU input

void SharpNSLS2::setGNode() {
  isGNode = true;
}

void SharpNSLS2::setChunks(int v){
     m_chunks = v;
}

// Recon Output

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getObject(){
  return m_engine->getObject();
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getProbe(){
  return m_engine->getProbe();
}

float SharpNSLS2::getObjectError(){
  return m_engine->getObjectError();
}

float SharpNSLS2::getProbeError(){
  return m_engine->getProbeError();
}

// SHARP internal containers

boost::multi_array<std::complex<float>, 3> & SharpNSLS2::getFrames(){
  return m_engine->getFrames();
}

boost::multi_array<std::complex<float>, 1> & SharpNSLS2::getFramesCorners(){
  return m_engine->getFramesCorners();
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getImageScale(){
  return m_engine->getImageScale();
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getIlluminatedArea(){
  return m_engine->getIlluminatedArea();
}

boost::multi_array<int, 1> & SharpNSLS2::getOverlapingFrames(){
  return m_engine->getOverlapingFrames();
}

boost::multi_array<int, 1> & SharpNSLS2::getOverlapingFramesIndex(){
  return m_engine->getOverlapingFramesIndex();
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getIlluminationNumerator(){
  return m_engine->getIlluminationNumerator();
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getIlluminationDenominator(){
  return m_engine->getIlluminationDenominator();
}

boost::multi_array<std::complex<float>, 3> & SharpNSLS2::getPrbObj(){
  return m_engine->getPrbObj();
}

boost::multi_array<std::complex<float>, 3> & SharpNSLS2::getTmp2(){
  return m_engine->getTmp2();
}

// MPI/GPU output

int SharpNSLS2::getRank(){
     return m_communicator->getRank();
}

// 

int SharpNSLS2::setArgs(int argc, char * argv[]){

  clean();

  // int argc_copy = argc;
  // Options* opt = Options::getOptions();
  // opt->parse_args(argc,argv);

  std::cout << "argv: " << argc << std::endl;
  for(int i=0; i < argc; i++){
  	  std::cout << i << " " << argv[i] << std::endl;
  }

  m_engine = new CudaEngineDM();

  // m_engine->setWrapAround(opt->wrapAround);

  if(isGNode) {
    m_communicator = new CommunicatorGNode(argc, argv, m_engine);
  } else {
    m_communicator = new CommunicatorMPI(argc, argv, m_engine);
  }

  bool result = true;

  // m_input_output = new InputOutput(argc_copy,argv, m_communicator);  
  // result = m_input_output->loadMetadata(opt->input_file.c_str());

  if(!result) {
    sharp_error("Error: failed to parse cxi file. exiting");
    return (1);
  }

  return 0;
}


int SharpNSLS2::init(){

    try{
      if(initSolver() == -1) {
	sharp_error("Solver/Engine has failed to initialize");
	clean();
	return -1;
      }
    }catch (std::exception& e){
      sharp_error("Solver/Engine has failed to initialize: %s\n", e.what());
      clean();
      exit(-1);
    }

    // Recon input

    m_engine->setAlpha(m_alpha);
    m_engine->setBeta(m_beta);
    m_engine->setStartUpdateProbe(m_start_update_probe);
    m_engine->setStartUpdateObject(m_start_update_object);
    m_engine->setAmpMax(m_amp_max);
    m_engine->setAmpMin(m_amp_min);
    m_engine->setPhaMax(m_pha_max);
    m_engine->setPhaMin(m_pha_min);

    // GPU input

    m_engine->setChunks(m_chunks);
    m_engine->init();
    
    return 0;
}

int SharpNSLS2::step(){
    return m_engine->step();
}

void SharpNSLS2::clean(){
    // if(m_solver) delete m_solver;
    if(m_strategy) delete m_strategy;
    if(m_input_output) delete m_input_output;
    if(m_engine) delete m_engine;
}

//

int SharpNSLS2::initSolver(){

  int res = -1;
  int argc = 0;
  char* argv[] = {};

  std::cout << "SharpNSLS2::initSolver()" << std::endl;

  m_engine = new CudaEngineDM();

  // m_engine->setWrapAround(opt->wrapAround);

  if(isGNode) {
    m_communicator = new CommunicatorGNode(argc, argv, m_engine);
  } else {
    m_communicator = new CommunicatorMPI(argc, argv, m_engine);
  }

  calculateReciprocalSize();

  // m_engine->setInputOutput(m_input_output);

  m_strategy = new Strategy(m_communicator);
  m_strategy->setTranslation(allTranslations());
  m_strategy->setFramesSize(framesSize());
  m_strategy->setReciprocalSize(reciprocalSize());
  m_strategy->calculateDecomposition();

  // frames
  // m_input_output->loadMyFrames(opt->input_file.c_str(), m_strategy->myFrames());
  
  loadMyFrames(m_strategy->myFrames()); // nm: update m_frames for multi-nodes
  m_engine->setTotalFrameCount(getTotalFrameCount());
  m_engine->setFrames(HostRange<float>(frames()));
  
  // m_engine->setReciprocalSize(m_input_output->reciprocalSize());
  m_engine->setReciprocalSize(reciprocalSize());

  // scan

  m_engine->setTranslations(HostRange<double>(translation()));
  m_engine->setAllTranslations(HostRange<double>(allTranslations()));

  // initial probe
  
  if(m_has_init_probe){
    // convert from std to cusp and m_engine->setIllumination
    // sharp_log("Setting initial probe");
    m_engine->setInitProbe(m_init_probe);
  } else {
    sharp_error("Initial probe is not defined");
    return res;
  }

  m_engine->setIlluminationMask(illumination_mask());
  m_engine->setIlluminationIntensities(illumination_intensities());

  // initial object

  if(m_has_init_object){
    // convert from std to cusp and m_engine->setImage
    // sharp_log("Setting initial object");
    m_engine->setInitObject(m_init_object);
  } else {
    sharp_error("Initial object is not defined");
    return res;
  }

  m_engine->setCommunicator(m_communicator);
  
  res = m_engine->initialize();

  return res;
}

// InputOutput interface


std::vector<int> SharpNSLS2::framesSize(){
  std::vector<int> size(3);
  size[0] = m_all_translations.shape()[0];
  ///TODO: check on this
  size[1] = m_frames_height; //m_frames.shape()[1];
  size[2] = m_frames_width; //m_frames.shape()[2];
  return size;
}

std::vector<double> SharpNSLS2::reciprocalSize() {
   return m_reciprocal_size;
}

boost::multi_array<double, 2> & SharpNSLS2::allTranslations(){
  return m_all_translations;
}

boost::multi_array<double, 2> & SharpNSLS2::translation(){
  return m_translation;
}

int SharpNSLS2::getTotalFrameCount() {
    return m_total_n_frames;
}

boost::multi_array<float, 3> & SharpNSLS2::frames(){
    return m_frames;
}

void SharpNSLS2::genMeanBackground(){
   m_mean_background.resize(boost::extents[m_frames_height][m_frames_width]);
   std::fill(m_mean_background.data(), m_mean_background.data()+
	     m_mean_background.num_elements(), 0);
}

void SharpNSLS2::loadMyFrames(const std::vector<int> & frames){

    genMeanBackground();
    
    /* shift the frames */
    boost::multi_array<float, 2> tmp;
    int n_frames = m_frames.shape()[0];
    int frames_h = m_frames.shape()[1];
    int frames_w = m_frames.shape()[2];
    tmp.resize(boost::extents[frames_h][frames_w]);
    for(int n = 0;n<n_frames;n++){
      for(int y = 0;y<frames_h;y++){
	for(int x = 0;x<frames_w;x++){
	  tmp[(y+frames_h/2)%frames_h][(x+frames_w/2)%frames_w] =  m_frames[n][y][x];
	}
      }
      for(int y = 0;y<frames_h;y++){	
	for(int x = 0;x<frames_w;x++){
	  m_frames[n][y][x] = tmp[y][x];
	}
      }
    }
    for(int y = 0;y<frames_h;y++){
      for(int x = 0;x<frames_w;x++){
	tmp[(y+frames_h/2)%frames_h][(x+frames_w/2)%frames_w] =  m_mean_background[y][x];
      }
    }
    for(int y = 0;y<frames_h;y++){	
      for(int x = 0;x<frames_w;x++){
	m_mean_background[y][x] = tmp[y][x];
      }
    }

    // Set the translations for this process
    m_translation.resize(boost::extents[frames.size()][m_all_translations.shape()[1]]);
    for(int i = 0; i < frames.size(); ++i) {
      m_translation[i] = m_all_translations[frames[i]];
    }
}

void  SharpNSLS2::calculateReciprocalSize(){

    // double h = 6.626070040e-34; // Js
    // double c = 2.99792458e8;    // m/s
    // double wavelength = h*c/energy; // m
    
    double wavelength = m_lambda_nm*1.e-9;
    double pixel_size = m_ccd_pixel_um*1.e-6;
    
    m_reciprocal_size[0] = pixel_size*m_frames_width/(m_z_m * wavelength);
    m_reciprocal_size[1] = pixel_size*m_frames_height/(m_z_m * wavelength);
    
    m_center[0] = m_frames_width/2.0;
    m_center[1] = m_frames_height/2.0;

    std::cout << "calculate reciprocal size: " <<  m_reciprocal_size[0] << ", " << m_reciprocal_size[1] << std::endl;
}

boost::multi_array<std::complex<float>, 2> & SharpNSLS2::illumination_mask(){ 
  return m_illumination_mask;
}

boost::multi_array<float, 2> & SharpNSLS2::illumination_intensities(){
  return m_illumination_intensities;
}


// depricated

int SharpNSLS2::run(){

    Options* opt = Options::getOptions();

    m_engine->iterate(opt->iterations);

    Counter::getCounter()->printTotals(m_communicator->getRank());

    return 0;
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getImage(){
  return m_engine->getImage();
}

void SharpNSLS2::writeImage(){
    char buffer[1024];
    sprintf(buffer,"run-%05d-", 0);
    // m_solver->writeImage(std::string(buffer));
}

std::string SharpNSLS2::getInputFile(){
   Options* opt = Options::getOptions();
   return opt->input_file;
}


