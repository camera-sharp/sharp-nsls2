
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Numpy REQUIRED)

include_directories(${NUMPY_INCLUDE_DIRS})
include_directories(${MPI_INCLUDE_PATH})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${INCLUDE_LIST})

  ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print sysconfig.get_python_lib(plat_specific=True, prefix='${CMAKE_INSTALL_PREFIX}')"
execute_process(
  COMMAND
  ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print sysconfig.get_python_lib(plat_specific=True)"
  OUTPUT_VARIABLE _PYTHON_INSTDIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

SET(PYTHON_INSTDIR ${_PYTHON_INSTDIR} CACHE PATH "Installation directory for python module.")

message("${PYTHON_INSTDIR}")
message("${CMAKE_INSTALL_PREFIX}")

FILE(GLOB sharpnsls2_headers "${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h")

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sharpnsls2_wrap.cu COMMAND ${SWIG_EXECUTABLE} -python -c++ -outcurrentdir ${SWIG_EXTRA_FLAGS} -I${CUDA_INCLUDE_DIRS} -I${CMAKE_CURRENT_SOURCE_DIR}/../include/ -o ${CMAKE_CURRENT_BINARY_DIR}/sharpnsls2_wrap.cu ${CMAKE_CURRENT_SOURCE_DIR}/sharpnsls2.i DEPENDS ${sharpnsls2_headers} ${CMAKE_CURRENT_SOURCE_DIR}/sharpnsls2.i)

INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH} ${PYTHON_NUMPY_INCLUDE_DIR})

CUDA_ADD_LIBRARY(_sharpnsls2 SHARED ${CMAKE_CURRENT_BINARY_DIR}/sharpnsls2_wrap.cu)

TARGET_LINK_LIBRARIES(_sharpnsls2 sharp-nsls2 ${PYTHON_LIBRARIES})
message("python libraries: ${PYTHON_INSTDIR}")

set_target_properties(
  _sharpnsls2
  PROPERTIES SOVERSION 1
  VERSION 1
  PREFIX ""
  SUFFIX ".so"
  INSTALL_NAME_DIR ${PYTHON_INSTDIR}
  )

INSTALL(TARGETS _sharpnsls2
  RUNTIME DESTINATION ${PYTHON_INSTDIR}
  LIBRARY DESTINATION ${PYTHON_INSTDIR}
  ARCHIVE DESTINATION ${PYTHON_INSTDIR}
  )

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/sharpnsls2.py ${CMAKE_CURRENT_SOURCE_DIR}/sharpnsls2/main.py ${CMAKE_CURRENT_SOURCE_DIR}/sharpnsls2/__init__.py DESTINATION ${PYTHON_INSTDIR}/sharpnsls2)
