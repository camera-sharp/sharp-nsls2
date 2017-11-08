#pragma once

#ifndef CUDA_ENGINE_NSLS2_H
#define CUDA_ENGINE_NSLS2_H

#include "cuda_engine.h"

class CudaEngineNSLS2: public CudaEngine
{
 public:

  /** Constructor */
  CudaEngineNSLS2();

 public:

  // Engine API

  virtual void iterate(int steps);

 public:

  // Fine-grained API

  void init();

  int step();

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

  int m_start_update_probe; // iteration number start updating probe, 2
  int m_start_update_object; // iteration number start updating object, 0

  float m_alpha; // general feedback parameter, 1.e-8
  float m_beta;  // espresso threshold coefficient, 0.9

  float m_amp_max;  // maximum object magnitude,  1.0
  float m_amp_min;  //  minimum object magnitude, 0.0
  float m_pha_max;  //  maximum object phase, pi/2
  float m_pha_min;  // minimum object phase, -pi/2 

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
