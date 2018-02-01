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

  /** detector distance, m */
  void setZ(float v);

  /** wavelength, nm */
  void setLambda(float v);

  /** ccd pixel size, um */
  void setPixelSize(float v);

 public:

  void setScan(const boost::multi_array<double, 2> & scan);

  void setFrames(const boost::multi_array<float, 3>& frames);

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

  float getChiError(); 

 public:

  // SHARP internal containers

  boost::multi_array<std::complex<float>, 3> & getFrames();

  boost::multi_array<std::complex<float>, 1> & getFramesCorners();

  boost::multi_array< int, 1> & getOverlapingFrames();

  boost::multi_array< int, 1> & getOverlapingFramesIndex();
  
  boost::multi_array<std::complex<float>, 2> & getIlluminatedArea();

  boost::multi_array<std::complex<float>, 2> & getImageScale();
 
  boost::multi_array<std::complex<float>, 2> & getIlluminationNumerator();

  boost::multi_array<std::complex<float>, 2> & getIlluminationDenominator();

  boost::multi_array<std::complex<float>, 3> & getPrbObj();

  boost::multi_array<std::complex<float>, 3> & getTmp2(); 

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

  // Solver::initialize

  int initSolver();

 protected:

  // InputOutput interface

  boost::multi_array<double, 2> & allTranslations();
  boost::multi_array<double, 2> & translation();

  // frames

  void loadMyFrames(const std::vector<int> & frames);
  void genMeanBackground();

  int getTotalFrameCount();
  boost::multi_array<float, 3> & frames();
  std::vector<int> framesSize();

  void calculateReciprocalSize();
  std::vector<double> reciprocalSize();

  // illumination

  boost::multi_array<std::complex<float>, 2> & illumination_mask();
  boost::multi_array<float, 2> & illumination_intensities();

 protected:

  CudaEngineDM* m_engine;
  Communicator* m_communicator;
  InputOutput* m_input_output;
  Strategy* m_strategy;
  //  Solver* m_solver;

 protected:

  // MPI/GPU parameters

  bool isGNode;
  int m_chunks;

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

  float m_z_m;               // detector distance, 0.0
  float m_lambda_nm;         // wavelength (nm), 0.0 
  float m_ccd_pixel_um;      // ccd_pixel_um, 0.0

 protected:

  // Input

  // translations defined by setScan

  bool m_has_scan; 
  boost::multi_array<double, 2> m_all_translations; 
  
  boost::multi_array<double, 2> m_translation;

  // initial probe defined by setInitProbe

  bool m_has_init_probe;
  boost::multi_array<std::complex<float>, 2> m_init_probe;

  // initial image defined by setInitObject

  bool m_has_init_object;
  boost::multi_array<std::complex<float>, 2> m_init_object;

  // frames

  int m_total_n_frames;
  int m_frames_width;
  int m_frames_height;

  bool m_has_frames;
  boost::multi_array<float, 3> m_frames;

  std::vector<float> m_center;

  // Size of the frames in reciprocal space
  std::vector<double> m_reciprocal_size;

 protected:

   // not used
   
   boost::multi_array<float, 2> m_mean_background;
   boost::multi_array<std::complex<float>, 2> m_illumination_mask;
   boost::multi_array<float, 2> m_illumination_intensities;

};

#endif
