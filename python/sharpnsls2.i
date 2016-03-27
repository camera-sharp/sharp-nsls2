%module(directors="1", threads="1") sharpnsls2

%{
#define SWIG_FILE_WITH_INIT
#include <dlfcn.h>
#include "SharpNSLS2.h"
%}

%init %{
  // This is here to avoid problems with missing symbols due to
  // the bad interactions between python and the crazy ways MPI
  // loads its libraries.
  // Check for example https://code.google.com/p/petsc4py/issues/detail?id=14
  dlopen("libmpi.so", RTLD_NOW|RTLD_GLOBAL|RTLD_NOLOAD);
%}


%include <std_string.i>
%include <std_vector.i>

namespace std {
  %template(IntVector) vector<int>;
  %template(FloatVector) vector<float>;
  %template(DoubleVector) vector<double>;
}

%include <argcargv.i>
%apply (int ARGC, char **ARGV) { (int argc, char *argv[]) }
%apply (int ARGC, char **ARGV) { (int argc, char **argv) }

/*****************************************/
/**                                     **/
/**          Start of typemaps          **/
/**                                     **/
/*****************************************/

/* Conversion from python integers to void * */
%typemap(in) void * {
  $1 = (void *)PyLong_AsVoidPtr($input);
}
/* This is necessary to allow the corresponding typemap(in)
   to be called for overloaded functions */
%typecheck(14) const int * {
  $1 = (PyInt_Check($input) || PyLong_Check($input)) ? 1 : 0;
}

/* Conversion from python integers to int *  */
%typemap(in) const int * dims{
  $1 = (int *)PyLong_AsVoidPtr($input);
}
/* This is necessary to allow the corresponding typemap(in)
   to be called for overloaded functions */
%typecheck(13) const int * {
  $1 = (PyInt_Check($input) || PyLong_Check($input)) ? 1 : 0;
}

/*****************************************/
/**                                     **/
/**          Exception handling         **/
/**                                     **/
/*****************************************/

%exception{
    try{
        $function
    }
    catch(Swig::DirectorException & e){
      PyErr_SetString(PyExc_RuntimeError, e.what());
      return NULL;
    }
}

%feature("director:except") {
  PyErr_Print();
  throw Swig::DirectorMethodException("Check traceback above");
}

/*****************************************/
/**                                     **/
/**      Start of API declaration       **/
/**                                     **/
/*****************************************/

%feature("director") SharpNSLS2; 

%apply float *INOUT { float * output_error };

%include "SharpNSLS2.h"
