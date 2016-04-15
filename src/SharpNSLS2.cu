#include <boost/assert.hpp>
#include <input_output.h>
#include <communicator.h>
#include <strategy.h>
#include <CudaEngineNSLS2.h>
#include <solver.h>
#include <string>
#include <unistd.h>
#include <options.h>
#include <exception>
#include "counter.h"

#include "SharpNSLS2.h"

SharpNSLS2::SharpNSLS2() {
  m_engine = 0;
  m_communicator = 0;
  m_input_output = 0;
  m_strategy = 0;
  m_solver = 0;
}

SharpNSLS2::~SharpNSLS2() {
  clean();
}

boost::multi_array<std::complex<float>, 2>& SharpNSLS2::getImage(){
  return m_engine->getImage();
}

int SharpNSLS2::init(int argc, char * argv[]){

  int argc_copy = argc;

  Options* opt = Options::getOptions();
  opt->parse_args(argc,argv);

  m_engine = new CudaEngineNSLS2();

  m_engine->setWrapAround(opt->wrapAround);

  m_communicator = new Communicator(argc, argv, m_engine);  

  m_input_output = new InputOutput(argc_copy,argv, m_communicator);  
  bool result = m_input_output->loadMetadata(opt->input_file.c_str());

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
 
  m_input_output->loadMyFrames(opt->input_file.c_str(), m_strategy->myFrames());

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

  m_engine->init();

  return 0;
}

int SharpNSLS2::run(){

    Options* opt = Options::getOptions();

    // m_solver->run(opt->iterations);
    for(int i = 0; i < opt->iterations; i++) {
       m_engine->step();
    }

   Counter::getCounter()->printTotals(m_communicator->getRank());

   return 0;
}

int SharpNSLS2::step(){
    return m_engine->step();
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
    if(m_communicator) delete m_communicator;
    if(m_engine) delete m_engine;

}

