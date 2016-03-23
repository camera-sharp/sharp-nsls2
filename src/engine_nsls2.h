#pragma once

#ifndef NSLS2_ENGINE_H
#define NSLS2_ENGINE_H

#include "sharp_thrust.h"
#include "cuda_engine.h"

class CudaEngineNSLS2: public CudaEngine
{
 public:

  /** Constructor */
  CudaEngineNSLS2();

  // Engine API

  virtual void iterate(int steps);

 protected:

  // recon_ptycho methods

  void cal_object_trans(const DeviceRange<cusp::complex<float> > & input_frames,
		   bool global_sync);

  void cal_probe_trans(const DeviceRange<cusp::complex<float> > & input_frames, 
		   const DeviceRange<cusp::complex<float> > & frames_object,
		   const DeviceRange<cusp::complex<float> > & frames_numerator,
		   const DeviceRange<cusp::complex<float> > & frames_denominator,
		   const DeviceRange<cusp::complex<float> > & prb_tmp);

  double cal_obj_error(const DeviceRange<cusp::complex<float> > & obj_old);

  double cal_prb_error(const DeviceRange<cusp::complex<float> > & prb_old);

  double cal_chi_error(const DeviceRange<cusp::complex<float> > & input_image, 
		       const DeviceRange<cusp::complex<float> > & tmp_frames);

 protected:

  // local versions of sharp methods/wrappers

  void calcDataProjector(const DeviceRange<cusp::complex<float> > & input_frames,
			 const DeviceRange<cusp::complex<float> > & output_frames,
			 float * output_residual = NULL);

  void calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_frames,
			     const DeviceRange<cusp::complex<float> > & input_image,
			     const DeviceRange<cusp::complex<float> > & output_frames,
			     float * output_residual = NULL);

  void calculateImageScale();

  int printDiagmostics(float data_residual, float overlap_residual);

  void printSummary(int success);

  double compareImageSolution();

 protected:

  // debugging 

  double cal_sol_error();

 public:

  // recon_ptycho parameters

  int m_start_update_probe;
  int m_start_update_object;

  float m_alpha; // 1.e-8
  float m_beta;  // 0.9

  float m_amp_max;  //  1.0
  float m_amp_min;  //  0.0
  float m_pha_max;  //  pi/2
  float m_pha_min;  // -pi/2 

 protected:

  // local sharp parameters

  DeviceRange<cusp::complex<float> >m_illuminated_area0;

  float m_data_tolerance;
  float m_overlap_tolerance;
  float m_solution_tolerance;

};

#endif
