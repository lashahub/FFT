cmake_minimum_required(VERSION 3.10)
project(FFT)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(FFT main.cpp utils.cpp utils.hpp)

target_link_libraries(FFT PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
