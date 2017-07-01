#pragma once

#ifndef COMMUNICATOR_GNODE_H
#define COMMUNICATOR_GNODE_H

#include "communicator_mpi.h"

class CommunicatorGNode: public CommunicatorMPI
{
 public:

  /** Constructor which receives command line arguments,
    to process MPI arguments, and an Engine to get
    the parameters of the problem. 
  */
  CommunicatorGNode(int argc, char ** argv, Engine * engine);

};

#endif
