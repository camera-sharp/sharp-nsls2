#pragma once

#ifndef CUDA_ENGINE_NSLS2_H
#define CUDA_ENGINE_NSLS2_H

#include "cuda_engine.h"

class CudaEngineNSLS2: public CudaEngine
{
 public:

  /** Constructor */
  CudaEngineNSLS2();

  // Engine API

  virtual void iterate(int steps);

 public:

  // New fine-grained API

  void init();

  int step();

 protected:

  // recon_ptycho methods

  void recon_dm_trans();

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

  void set_object_constraints();

 protected:

  // local versions of sharp methods/wrappers

  void calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_frames,
			     const DeviceRange<cusp::complex<float> > & input_image,
			     const DeviceRange<cusp::complex<float> > & output_frames,
			     float * output_residual = NULL);

  void calculateImageScale();

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

    DeviceRange<cusp::complex<float> > m_image_old; 
    DeviceRange<cusp::complex<float> > m_prb_old; 
    DeviceRange<cusp::complex<float> > m_prb_tmp; 

    // Large GPU arrays, with size equal to number of frames 
 
    thrust::device_vector<cusp::complex<float> > m_prb_obj;
    thrust::device_vector<cusp::complex<float> > m_tmp;
    thrust::device_vector<cusp::complex<float> > m_tmp2;

};

#endif
