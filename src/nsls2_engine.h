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

  void iterate(int steps);

};

#endif
