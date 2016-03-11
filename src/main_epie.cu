#include <boost/assert.hpp>
#include <input_output.h>
#include <communicator.h>
#include <strategy.h>
#include <engine_epie.h>
#include <solver.h>
#include <string>
#include <unistd.h>
#include <options.h>
#include <exception>
#include "counter.h"

int main(int argc, char ** argv){
  int argc_copy = argc;

  /*! parse options */
  Options * opt = Options::getOptions();
  opt->parse_args(argc,argv);

  /*! initialize engine */
  CudaEngineEPIE engine;
  engine.setWrapAround(opt->wrapAround);
  Communicator communicator(argc, argv,&engine);  

  InputOutput input_output(argc_copy,argv,&communicator);  
  bool result = input_output.loadMetadata(opt->input_file.c_str());

  if(!result) {
    sharp_error("Error: failed to parse cxi file. exiting");
    return (1);
  }

  engine.setInputOutput(&input_output);

  Strategy strategy(&communicator);
  strategy.setTranslation(input_output.allTranslations());
  strategy.setFramesSize(input_output.framesSize());
  strategy.setReciprocalSize(input_output.reciprocalSize());
  strategy.calculateDecomposition();
 
  input_output.loadMyFrames(opt->input_file.c_str(),strategy.myFrames());

  for(int i =0;i<opt->n_reconstructions;i++){
    char buffer[1024];
    sprintf(buffer,"run-%05d-",i);
    Solver solver(&engine,&communicator,&input_output,&strategy);

    try{
      if(solver.initialize() == -1) {
	sharp_error("Solver/Engine has failed to initialize");
	return -1;
      }
    }catch (std::exception& e){
      sharp_error("Solver/Engine has failed to initialize: %s\n", e.what());
      exit(-1);
    }
    solver.run(opt->iterations);
    solver.writeImage(std::string(buffer));
  }

  Counter::getCounter()->printTotals(communicator.getRank());
  return 0;  
}
