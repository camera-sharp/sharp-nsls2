#pragma once

#ifndef EPIE_ENGINE_H
#define EPIE_ENGINE_H

#include "sharp_thrust.h"
#include "cuda_engine.h"

class CudaEngineEPIE: public CudaEngine
{
 public:

  /** Constructor */
  CudaEngineEPIE();

  virtual void iterate(int steps);

 protected:

  void calcDelImage(const DeviceRange<cusp::complex<float> > & input_frames,
                    const DeviceRange<cusp::complex<float> > & output_image,
		    bool global_sync);

  void calcDelProbe(const DeviceRange<cusp::complex<float> > & input_frames, 
		      const DeviceRange<cusp::complex<float> > & frames_object,
		      const DeviceRange<cusp::complex<float> > & frames_numerator,
		      const DeviceRange<cusp::complex<float> > & frames_denominator);

  void calcDataProjector(const DeviceRange<cusp::complex<float> > & input_frames,
			 const DeviceRange<cusp::complex<float> > & output_frames,
			 float * output_residual = NULL);

  void calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_frames,
			     const DeviceRange<cusp::complex<float> > & input_image,
			     const DeviceRange<cusp::complex<float> > & output_frames,
			     float * output_residual = NULL);

 public:

  int printDiagmostics(float data_residual, float overlap_residual);

  void printSummary(int success);

 protected:

  thrust::device_vector<cusp::complex<float> > m_del_image;
  thrust::device_vector<cusp::complex<float> > m_del_illumination;

  float m_data_tolerance;
  float m_overlap_tolerance;
  float m_solution_tolerance;

};

#endif
