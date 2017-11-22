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

  /** set the SHARP variables with command arguments */
  int setArgs(int argc, char * argv[]);

 public:

  // Recon Input API

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

  void setInitObject(const boost::multi_array<std::complex<float>, 2> & object);

  void setInitProbe(const boost::multi_array<std::complex<float>, 2> & probe);
  
  public:

  // MPI/GPU Input

  void setGNode();

  void setChunks(int chunks);

 public:

  // Recon Output API

  boost::multi_array<std::complex<float>, 2> & getObject();

  boost::multi_array<std::complex<float>, 2> & getProbe();

  float getObjectError();

  float getProbeError();

  public:

  // MPI/GPU Output

  int getRank();

 public:

  /** Initialize engine */
  int init();

  /** Run one iteration */
  int step();

 public:

  // depricated

  int run(); 

  boost::multi_array<std::complex<float>, 2>& getImage();

  void writeImage();

  std::string getInputFile();

 protected:

   void clean();

 protected:

  CudaEngineDM* m_engine;
  Communicator* m_communicator;
  InputOutput* m_input_output;
  Strategy* m_strategy;
  Solver* m_solver;

 protected:

  // Recon input parameters

  int m_start_update_probe;  // iteration number start updating probe, 2
  int m_start_update_object; // iteration number start updating object, 0

  float m_alpha; // espresso threshold coefficient, 1.e-8
  float m_beta;  // general feedback parameter, 0.9

  float m_amp_max;  //  maximum object magnitude, 1.0
  float m_amp_min;  //  minimum object magnitude, 0.0
  float m_pha_max;  //  maximum object phase, pi/2
  float m_pha_min;  // minimum object phase, -pi/2

  bool m_has_init_probe;
  boost::multi_array<std::complex<float>, 2> m_init_probe;

  bool m_has_init_object;
  boost::multi_array<std::complex<float>, 2> m_init_object;

  // MPI/GPU parameters

  bool isGNode;

  int m_chunks;

  // SHARP input parameters




};

#endif
