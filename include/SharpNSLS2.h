#pragma once

#ifndef SHARP_NSLS2_H
#define SHARP_NSLS2_H

#include <string>
#include "boost/multi_array.hpp"

class CudaEngineNSLS2;
class Communicator;
class InputOutput;
class Strategy;
class Solver;

class SharpNSLS2 {

 public:

  SharpNSLS2();

  ~SharpNSLS2();

 public:

  int init(int argc, char * argv[]);

  int run();

  int step();

  boost::multi_array<std::complex<float>, 2>& getImage();

  void writeImage();

  void clean();

  std::string getInputFile();

 protected:

  CudaEngineNSLS2* m_engine;
  Communicator* m_communicator;
  InputOutput* m_input_output;
  Strategy* m_strategy;
  Solver* m_solver;

};

#endif
