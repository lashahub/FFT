cmake_minimum_required(VERSION 3.10)
project(FFT)

set(CMAKE_CXX_STANDARD 17)

add_executable(FFT main.cpp utils.cpp utils.hpp)
