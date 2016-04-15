#include <SharpNSLS2.h>

int main(int argc, char ** argv){

  SharpNSLS2 sharpNSLS2;

  sharpNSLS2.init(argc,argv);

  sharpNSLS2.run();

  sharpNSLS2.writeImage();

  return 0;  
}
