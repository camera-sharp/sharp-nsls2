# 
# Try to find ARRAYFIRE  library  
# (see https://github.com/arrayfire/arrayfire)
# Once run this will define: 
# 
# ARRAYFIRE_FOUND
# ARRAYFIRE_INCLUDE_DIR 
# ARRAYFIRE_LIBRARIES
# ARRAYFIRE_LINK_DIRECTORIES
#
# Filipe RNC Maia 06/2015
# --------------------------------
SET(ARRAYFIRE_DIR "" CACHE STRING "ARRAYFIRE installation directory")

FIND_PATH(ARRAYFIRE_INCLUDE_DIR arrayfire.h
   /usr/include
   /usr/local/include
   $ENV{C_INCLUDE_PATH}
   $ENV{CPLUS_INCLUDE_PATH}
   ${ARRAYFIRE_DIR}/include
   ${ARRAYFIRE_INCLUDE_DIR}
 )



SET(ARRAYFIRE_POSSIBLE_LIBRARY_PATH
  ${ARRAYFIRE_DIR}/lib
  /usr/lib
  /usr/local/lib
  /usr/lib64
  /usr/local/lib64
  $ENV{LD_LIBRARY_PATH}
  $ENV{LIBRARY_PATH}
)

  
FIND_LIBRARY(ARRAYFIRE_CUDA_LIBRARY
  NAMES afcuda
  PATHS 
  ${ARRAYFIRE_POSSIBLE_LIBRARY_PATH}
  )


# --------------------------------
# select one of the above
# default: 
IF (ARRAYFIRE_CUDA_LIBRARY)
  SET(ARRAYFIRE_LIBRARIES ${ARRAYFIRE_CUDA_LIBRARY})
ENDIF (ARRAYFIRE_CUDA_LIBRARY)

# --------------------------------

IF(ARRAYFIRE_LIBRARIES)
  IF (ARRAYFIRE_INCLUDE_DIR)

    # OK, found all we need
    SET(ARRAYFIRE_FOUND TRUE)
    GET_FILENAME_COMPONENT(ARRAYFIRE_LINK_DIRECTORIES ${ARRAYFIRE_LIBRARIES} PATH)
    
  ELSE (ARRAYFIRE_INCLUDE_DIR)
    MESSAGE("ARRAYFIRE include dir not found. Set ARRAYFIRE_DIR to find it.")
  ENDIF(ARRAYFIRE_INCLUDE_DIR)
ELSE(ARRAYFIRE_LIBRARIES)
  MESSAGE("ARRAYFIRE lib not found. Set ARRAYFIRE_DIR to find it.")
ENDIF(ARRAYFIRE_LIBRARIES)


MARK_AS_ADVANCED(
  ARRAYFIRE_INCLUDE_DIR
  ARRAYFIRE_LIBRARIES
  ARRAYFIRE_CUDA_LIBRARY
  ARRAYFIRE_LINK_DIRECTORIES
)

include( FindPackageHandleStandardArgs ) 
find_package_handle_standard_args( ARRAYFIRE 
    REQUIRED_VARS 
        ARRAYFIRE_LIBRARIES 
) 

