cmake_minimum_required(VERSION 3.20)

# set(CMAKE_Fortran_COMPILER ifort)

project(FilterProject LANGUAGES C CXX Fortran)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "-march=native -mtune=native")
set(CMAKE_Fortran_FLAGS "-O3 -xHost")

add_executable(program main.cpp filter.F90)
set_source_files_properties(filter.F90 PROPERTIES LANGUAGE Fortran)
