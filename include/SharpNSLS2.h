#pragma once

#ifndef SHARP_NSLS2_H
#define SHARP_NSLS2_H

#include <string>
#include "boost/multi_array.hpp"

class CudaEngineDM;
class Communicator;
class InputOutput;
class Strategy;
class Solver;

class SharpNSLS2 {

 public:

  /** Constructor */
  SharpNSLS2();

  /** Destructor */
  ~SharpNSLS2();

 public:

  void setGNode();

 public:

  /** set the SHARP variables with command arguments */
  int setArgs(int argc, char * argv[]);

 public:

  // Recon API

  /** general feedback parameter */
  void setBeta(float v);

  /** espresso threshold coefficient */
  void setAlpha(float v);

  /** iteration number start updating probe */
  void setStartUpdateProbe(int v);

  /** iteration number start updating object */
  void setStartUpdateObject(int v);

  /** maximum object magnitude */
  void setAmpMax(float v);

  /** minimum object magnitude */
  void setAmpMin(float v);

  /** maximum object phase */
  void setPhaMax(float v);

  /** minimum object phase */
  void setPhaMin(float v);

 public:

    void setChunks(int chunks);

 public:

  /** Initialize engine */
  int init();

  /** Run all iterations */
  int run();

  /** Run one iteration */
  int step();

  boost::multi_array<std::complex<float>, 2>& getImage();

  void writeImage();

  void clean();

  std::string getInputFile();

 protected:

  CudaEngineDM* m_engine;
  Communicator* m_communicator;
  InputOutput* m_input_output;
  Strategy* m_strategy;
  Solver* m_solver;

 protected:

  bool isGNode;

};

#endif
