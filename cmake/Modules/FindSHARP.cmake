# 
# Try to find CAMERA-SHARP  library  
# (see http://www.camera.lbl.gov/#!ptychography/dtl7e)
# Once run this will define: 
# 
# SHARP_FOUND
# SHARP_INCLUDE_DIR 
# SHARP_LIBRARIES
# SHARP_LINK_DIRECTORIES
#
# N. Malitsky, March 2016
# --------------------------------

FIND_PATH(SHARP_INCLUDE_DIR sharp.h
   /usr/include
   /usr/local/include
   $ENV{C_INCLUDE_PATH}
   $ENV{CPLUS_INCLUDE_PATH}
   ${SHARP_DIR}/include
   ${SHARP_INCLUDE_DIR}
 )

SET(SHARP_POSSIBLE_LIBRARY_PATH
  ${SHARP_DIR}/lib
  /usr/lib
  /usr/local/lib
  /usr/lib64
  /usr/local/lib64
  $ENV{LD_LIBRARY_PATH}
  $ENV{LIBRARY_PATH}
)

  
FIND_LIBRARY(SHARP_CUDA_LIBRARY
  NAMES sharp
  PATHS 
  ${ARRAYFIRE_POSSIBLE_LIBRARY_PATH}
  )


# --------------------------------
# select one of the above
# default: 
IF (SHARP_CUDA_LIBRARY)
  SET(SHARP_LIBRARIES ${SHARP_CUDA_LIBRARY})
ENDIF (SHARP_CUDA_LIBRARY)

# --------------------------------

IF(SHARP_LIBRARIES)
  IF (SHARP_INCLUDE_DIR)
    MESSAGE("-- Found SHARP: ${SHARP_DIR}")
    # OK, found all we need
    SET(SHARP_FOUND TRUE)
    GET_FILENAME_COMPONENT(SHARP_LINK_DIRECTORIES ${SHARP_LIBRARIES} PATH)
    
  ELSE (SHARP_INCLUDE_DIR)
    MESSAGE("SHARP include dir not found. Set SHARP_DIR to find it.")
  ENDIF(SHARP_INCLUDE_DIR)
ELSE(SHARP_LIBRARIES)
  MESSAGE("SHARP lib not found. Set SHARP_DIR to find it.")
ENDIF(SHARP_LIBRARIES)


MARK_AS_ADVANCED(
  SHARP_INCLUDE_DIR
  SHARP_LIBRARIES
  SHARP_CUDA_LIBRARY
  SHARP_LINK_DIRECTORIES
)

include( FindPackageHandleStandardArgs ) 
find_package_handle_standard_args( SHARP 
    REQUIRED_VARS 
        SHARP_LIBRARIES 
) 

