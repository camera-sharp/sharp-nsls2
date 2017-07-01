cmake_minimum_required(VERSION 2.8)

if(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW) # CMake 3.0
endif()

if(POLICY CMP0025)
  cmake_policy(SET CMP0025 NEW) # CMake 3.0
endif()

PROJECT (sharp-nsls2)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

include_directories("include")

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
SET(INCLUDE_LIST ${CMAKE_CURRENT_SOURCE_DIR}/include/)


if(NOT ENV{HDF5_ROOT})
    if(HDF5_DIR)
        set(ENV{HDF5_ROOT} ${HDF5_DIR})
    elseif(HDF5_ROOT)
        set(ENV{HDF5_ROOT} ${HDF5_ROOT})
    endif()
endif()

# HDF5
find_package(HDF5 REQUIRED)
mark_as_advanced(CLEAR HDF5_hdf5_LIBRARY_RELEASE)
mark_as_advanced(CLEAR HDF5_C_INCLUDE_DIR)
mark_as_advanced(HDF5_DIR)

# BOOST
find_package(Boost REQUIRED QUIET)
# find_package(CUB REQUIRED)

# FFTW3
find_package(FFTW3 REQUIRED)

# Find CUSP and Thrust
find_package(CUDA REQUIRED)
mark_as_advanced(CUDA_BUILD_EMULATION)
mark_as_advanced(CUDA_SDK_ROOT_DIR)
mark_as_advanced(CUDA_SEPARABLE_COMPILATION)
mark_as_advanced(CUDA_VERBOSE_BUILD)
mark_as_advanced(CUDA_BUILD_CUBIN)
mark_as_advanced(CUDA_HOST_COMPILER)

# find_package(CUSP REQUIRED)
# mark_as_advanced(CUSP_INCLUDE_DIR)

find_package(THRUST REQUIRED)
mark_as_advanced(THRUST_INCLUDE_DIR)

if(NOT SHARP_DIR)
  set(SHARP_DIR $ENV{SHARP_HOME})
endif()

find_package(SHARP)
include_directories(${SHARP_INCLUDE_DIR})

SET(BUILD_WITH_OPENMP OFF CACHE BOOL "If ON build sharp_openmp, the OpenMP version of sharp.")
mark_as_advanced(BUILD_WITH_OPENMP)

if(BUILD_DOCS)
  find_package(Doxygen)
endif()

include(InstallRequiredSystemLibraries)

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NSLS-II extension of the CAMERA-SHARP ptychography code")
SET(CPACK_PACKAGE_VENDOR "Brookhaven National Laboratory")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
# SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_VERSION_MAJOR "1")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake
${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")

IF(WIN32 AND NOT UNIX)
ELSE(WIN32 AND NOT UNIX)
  SET(CPACK_STRIP_FILES "bin/sharp-nsls2")
  SET(CPACK_SOURCE_STRIP_FILES "")
ENDIF(WIN32 AND NOT UNIX)
SET(CPACK_PACKAGE_EXECUTABLES "sharp-nsls2" "sharp-nsls2")
INCLUDE(CPack)

include_directories(${Boost_INCLUDE_DIRS})

FIND_PACKAGE(MPI REQUIRED)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Necessary due to CUDA compatibility problems
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  # The CUDA module thinks cc is gcc and passes dumpspecs.
  # This is a workaround
  if(CUDA_HOST_COMPILER STREQUAL "/usr/bin/cc")
    SET(CUDA_HOST_COMPILER "/usr/bin/clang")
  endif()
endif()

# Default to sm_30 and sm_35
if(NOT CUDA_NVCC_FLAGS)
  SET(CUDA_NVCC_FLAGS "-gencode=arch=compute_30,code=compute_30 -gencode=arch=compute_35,code=compute_35 -DBOOST_DISABLE_ASSERTS")
endif()

if(TIMINGS)
  ADD_DEFINITIONS(-DTIMINGS)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DTIMINGS")
endif(TIMINGS)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "release")
endif()

LIST(APPEND INCLUDE_LIST ${MPI_INCLUDE_PATH})
LIST(APPEND LIBRARY_LIST ${MPI_LIBRARIES})

LIST(APPEND INCLUDE_LIST ${HDF5_INCLUDE_DIR})
LIST(APPEND LIBRARY_LIST ${HDF5_LIBRARIES})

LIST(APPEND INCLUDE_LIST ${FFTW3_INCLUDE_DIR})
LIST(APPEND LIBRARY_LIST ${FFTW3_LIBRARIES})

LIST(APPEND INCLUDE_LIST ${CUSP_INCLUDE_DIR})
LIST(APPEND INCLUDE_LIST ${CUB_INCLUDE_DIR})

LIST(APPEND INCLUDE_LIST ${SHARP_INCLUDE_DIR})
LIST(APPEND LIBRARY_LIST ${SHARP_LIBRARIES})


ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(include)

option ( ENABLE_PYTHON_BINDINGS "Enable Python bindings via Swig" ON)
IF(ENABLE_PYTHON_BINDINGS)
  ADD_SUBDIRECTORY(python)
ENDIF(ENABLE_PYTHON_BINDINGS)

# target etags/tags
IF (UNIX OR APPLE)
  ADD_CUSTOM_TARGET(tags etags --members --declarations  `find .. -name *.cc -or -name *.hh -or -name *.cpp -or -name *.h -or -name *.c -or -name *.f -or -name *.cu`)
  ADD_CUSTOM_TARGET(etags DEPENDS tags)
ENDIF (UNIX OR APPLE)



