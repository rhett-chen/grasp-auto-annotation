cmake_minimum_required(VERSION 2.6)
project("sdf_gen")

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries

if(NOT CMAKE_BUILD_TYPE)
   set(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
       FORCE)
endif()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

#These flags might not work on every system, especially the release flags, comment out as needed
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

#checks if VTK is available
find_package(VTK QUIET COMPONENTS vtkIOXML)
if(VTK_FOUND)
  set(HAVE_VTK 1)
endif()

#Generates config.h replacing expressions in config.h.in with actual values
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h"
)

add_executable(${PROJECT_NAME} src/main.cpp src/makelevelset3.cpp)

if(VTK_FOUND)
    include(${VTK_USE_FILE})
  target_link_libraries(${PROJECT_NAME} ${VTK_LIBRARIES})
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
install(PROGRAMS scripts/sdf_tools DESTINATION bin)
