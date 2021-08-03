cmake_minimum_required(VERSION 3.15)
project(ERE_exampe)

if(NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB SRCFILES src/*.cpp include/*.h)
add_subdirectory(example)