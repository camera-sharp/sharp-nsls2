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

#include "SharpNSLS2.h"
#include "CommunicatorGNode.h"

SharpNSLS2::SharpNSLS2() {
  m_engine = 0;
  m_communicator = 0;
  m_input_output = 0;
  m_strategy = 0;
  m_solver = 0;

  isGNode = false;

}

SharpNSLS2::~SharpNSLS2() {
  clean();
  if(m_communicator) delete m_communicator;
}

// Recon API

void SharpNSLS2::setAlpha(float v){
     m_engine->setAlpha(v);
}

void SharpNSLS2::setBeta(float v){
     m_engine->setBeta(v);
}

void SharpNSLS2::setStartUpdateProbe(int v){
     m_engine->setStartUpdateProbe(v);
}

void SharpNSLS2::setStartUpdateObject(int v){
     m_engine->setStartUpdateObject(v);
}

void SharpNSLS2::setAmpMax(float v){
     m_engine->setAmpMax(v);
}

void SharpNSLS2::setAmpMin(float v){
     m_engine->setAmpMin(v);
}

void SharpNSLS2::setPhaMax(float v){
     m_engine->setPhaMax(v);
}

void SharpNSLS2::setPhaMin(float v){
     m_engine->setPhaMin(v);
}

// 

int SharpNSLS2::setArgs(int argc, char * argv[]){

  clock_t start_timer = clock(); //Start 

  clean();

  double clean_timer = clock();
  double diff =  (clean_timer - start_timer)/(double) CLOCKS_PER_SEC; 
  // std::cout << "SharpNSLS2::init, clean, time: " << diff << std::endl;

  int argc_copy = argc;

  Options* opt = Options::getOptions();
  opt->parse_args(argc,argv);

  m_engine = new CudaEngineDM();

  m_engine->setWrapAround(opt->wrapAround);

  if(isGNode) {
    m_communicator = new CommunicatorGNode(argc, argv, m_engine);
  } else {
    m_communicator = new CommunicatorMPI(argc, argv, m_engine);
  }

  double comm_timer = clock();
  diff =  (comm_timer - clean_timer)/(double) CLOCKS_PER_SEC; 

  // std::cout << "SharpNSLS2::init, communicator, time: " << diff << std::endl;

  m_input_output = new InputOutput(argc_copy,argv, m_communicator);  
  bool result = m_input_output->loadMetadata(opt->input_file.c_str());

  double meta_timer = clock();
  diff = (meta_timer - comm_timer)/(double) CLOCKS_PER_SEC;

  // std::cout << "SharpNSLS2::init, loading metadata, time: " << diff << std::endl;

  if(!result) {
    sharp_error("Error: failed to parse cxi file. exiting");
    return (1);
  }

  m_engine->setInputOutput(m_input_output);

  m_strategy = new Strategy(m_communicator);
  m_strategy->setTranslation(m_input_output->allTranslations());
  m_strategy->setFramesSize(m_input_output->framesSize());
  m_strategy->setReciprocalSize(m_input_output->reciprocalSize());
  m_strategy->calculateDecomposition();

  double strategy_timer = clock();
  diff = (strategy_timer - meta_timer)/(double) CLOCKS_PER_SEC;

  // std::cout << "SharpNSLS2::init, strategy, time: " << diff << std::endl;
 
  m_input_output->loadMyFrames(opt->input_file.c_str(), m_strategy->myFrames());

  double frames_timer = clock();
  diff = (frames_timer - strategy_timer)/(double) CLOCKS_PER_SEC;

  // std::cout << "SharpNSLS2::init, loading frames, time: " << diff << std::endl;

  m_solver = new Solver(m_engine, m_communicator, m_input_output, m_strategy);

    try{
      if(m_solver->initialize() == -1) {
	sharp_error("Solver/Engine has failed to initialize");
	clean();
	return -1;
      }
    }catch (std::exception& e){
      sharp_error("Solver/Engine has failed to initialize: %s\n", e.what());
      clean();
      exit(-1);
    }

  diff = (clock() - frames_timer) / (double) CLOCKS_PER_SEC; 
  // std::cout << "SharpNSLS2::init, solver, time: " << diff << std::endl;

  return 0;
}

//

int SharpNSLS2::init(){
    m_engine->init();
    return 0;
}

int SharpNSLS2::run(){

    Options* opt = Options::getOptions();

    m_engine->iterate(opt->iterations);

    Counter::getCounter()->printTotals(m_communicator->getRank());

    return 0;
}

int SharpNSLS2::step(){
    return m_engine->step();
}

void SharpNSLS2::setGNode() {
  isGNode = true;
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getImage(){
  return m_engine->getImage();
}

void SharpNSLS2::writeImage(){
    char buffer[1024];
    sprintf(buffer,"run-%05d-", 0);
    m_solver->writeImage(std::string(buffer));
}

std::string SharpNSLS2::getInputFile(){
   Options* opt = Options::getOptions();
   return opt->input_file;
}

void SharpNSLS2::clean(){
    if(m_solver) delete m_solver;
    if(m_strategy) delete m_strategy;
    if(m_input_output) delete m_input_output;
    if(m_engine) delete m_engine;
}

