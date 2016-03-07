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

  virtual void iterate(int steps);

 protected:

  void calcImage(const DeviceRange<cusp::complex<float> > & input_frames,
		 const DeviceRange<cusp::complex<float> > & output_image,
		 bool global_sync);

  void calcOverlapProjection(const DeviceRange<cusp::complex<float> > & input_frames,
			     const DeviceRange<cusp::complex<float> > & input_image,
			     const DeviceRange<cusp::complex<float> > & output_frames,
			     float * output_residual = NULL);

  int  printDiagmostics(float data_residual, float overlap_residual);

  void printSummary(int success);

 protected:

  float m_data_tolerance;
  float m_overlap_tolerance;
  float m_solution_tolerance;

};

#endif
