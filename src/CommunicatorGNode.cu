#include <boost/assert.hpp>
#include <input_output.h>
#include <strategy.h>
#include <solver.h>
#include <string>
#include <unistd.h>
#include <options.h>
#include <exception>
#include "counter.h"

#include "CommunicatorGNode.h"

CommunicatorGNode::CommunicatorGNode(int argc, char ** argv, Engine * engine)
  : CommunicatorMPI(argc, argv, engine) {

  int rank = getRank();

  int gpuID;
  cudaSetDevice(rank);
  cudaGetDevice(&gpuID);

  std::cout << "CommunicatorGNode, rank: " << rank << std::endl;
  std::cout << "GPU id: " << gpuID << ", in rank: " << rank << std::endl;

}