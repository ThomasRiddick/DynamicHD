cmake_minimum_required (VERSION 3.0)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-stdlib=libc++ -std=gnu++11")
project(DynamicHD CXX)
file (GLOB SOURCE_AND_HEADER_FILES "Dynamic_HD_Cpp_Code/src/gtest/gtest-all.cc"
      "Dynamic_HD_Cpp_Code/src/algorithms/*" "Dynamic_HD_Cpp_Code/src/base/*"
      "Dynamic_HD_Cpp_Code/src/drivers/*" "Dynamic_HD_Cpp_Code/src/testing/*")
add_executable(Dynamic_HD_Cpp_Exec  "Dynamic_HD_Cpp_Code/src/gtest/gtest_main.cc" ${SOURCE_AND_HEADER_FILES})
include_directories("Dynamic_HD_Cpp_Code/include" "Dynamic_HD_Cpp_Code/src" "Dynamic_HD_Cpp_Code")
