cmake_minimum_required(VERSION 3.10)
project(FFT)

set(CMAKE_CXX_STANDARD 17)

add_executable(FFT main.cpp bmp.cpp bmp.hpp)

find_package(OpenMP REQUIRED)
if (OpenMP_CXX_FOUND)
    target_link_libraries(FFT PUBLIC OpenMP::OpenMP_CXX)
endif ()
