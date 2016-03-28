#pragma once

#ifndef SHARP_NSLS2_H
#define SHARP_NSLS2_H

#include <string>

class SharpNSLS2 {

 public:

  SharpNSLS2();

  int run(int argc, char * argv[]);

  std::string getInputFile();


};

#endif
