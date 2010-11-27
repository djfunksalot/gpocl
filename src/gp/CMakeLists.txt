# $Id: CMakeLists.txt 353 2010-07-31 09:34:09Z daaugusto $
#

CMAKE_MINIMUM_REQUIRED( VERSION 2.6 )

# This CMakeLists was designed to work with the policies of cmake 2.8.1
CMAKE_POLICY( VERSION 2.8.1 )

SET( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/")

# Configure OpenCL libraries
FIND_PACKAGE( OpenCL REQUIRED )

MARK_AS_ADVANCED( CLEAR OPENCL_INCLUDE_DIRS )
MARK_AS_ADVANCED( CLEAR OPENCL_LIBRARIES )

IF( OPENCL_INCLUDE_DIRS )
  INCLUDE_DIRECTORIES( ${OPENCL_INCLUDE_DIRS} )
ELSE( OPENCL_INCLUDE_DIRS )
  MESSAGE( SEND_ERROR ">  Could NOT find OPENCL includes." )
ENDIF( OPENCL_INCLUDE_DIRS )

IF( OPENCL_LIBRARIES )
  LINK_LIBRARIES( ${OPENCL_LIBRARIES} )
ELSE( OPENCL_LIBRARIES )
  MESSAGE( SEND_ERROR ">  Could NOT find OPENCL library." )
ENDIF( OPENCL_LIBRARIES )

IF( OPENCL_HAS_CPP_BINDINGS )
	MESSAGE( ">  OpenCL has C++ bindings. Full include is: " ${OPENCL_INCLUDE_DIRS} )
ELSE( OPENCL_HAS_CPP_BINDINGS )
	MESSAGE( ">  No OpenCL C++ bindings found" )
ENDIF( OPENCL_HAS_CPP_BINDINGS )

ADD_EXECUTABLE( gpocl gpocl.cc Params.cc GP.cc Primitives.cc CPU.cc GPU.cc )

# Link the executable to the GP and OpenCL library.
TARGET_LINK_LIBRARIES( gpocl Util ${OPENCL_LIBRARIES} )

# Copy to the binary directory all the individual OpenCL kernels (they are compiled on-the-fly)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/kernels/common.cl ${CMAKE_BINARY_DIR}/kernels/common.cl COPYONLY)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/kernels/cpu.cl ${CMAKE_BINARY_DIR}/kernels/cpu.cl COPYONLY)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpu_ppcu.cl ${CMAKE_BINARY_DIR}/kernels/gpu_ppcu.cl COPYONLY)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpu_ppce.cl ${CMAKE_BINARY_DIR}/kernels/gpu_ppce.cl COPYONLY)
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpu_fpi.cl ${CMAKE_BINARY_DIR}/kernels/gpu_fpi.cl COPYONLY)
