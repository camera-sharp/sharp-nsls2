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

  void updateImage(const DeviceRange<cusp::complex<float> > & input_frames,
		   bool global_sync);

  void updateProbe(const DeviceRange<cusp::complex<float> > & input_frames, 
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

  double compareImageSolution();

 public:

  int printDiagmostics(float data_residual, float overlap_residual);

  void printSummary(int success);

 protected:

  DeviceRange<cusp::complex<float> >m_illuminated_area0;

  float m_data_tolerance;
  float m_overlap_tolerance;
  float m_solution_tolerance;

};

#endif
