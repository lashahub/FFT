# CSE305 Project - Parallel Fast Fourier Transform

## Compilation Instructions

To compile this project, you will need to follow these steps:

1. Create a new directory for the build. This is a good practice as it helps to avoid building in the source directory. Use the following command to create a directory named "build":
    ```
    mkdir build
    ```
2. Change to the new directory:
    ```
    cd build
    ```
3. Run CMake with the `-DCMAKE_BUILD_TYPE=Release` option to specify a release build. The `..` at the end of the command refers to the parent directory, which should be the location of the source code:
    ```
    cmake -DCMAKE_BUILD_TYPE=Release ..
    ```
4. Once the makefile has been generated without errors, compile the project using `make`:
    ```
    make
    ```
5. To run the program, tye:
    ```
    ./FFT
    ```
    By default it runs all tests cases.

Following these steps will generate your executables and/or libraries in the `build` directory, built in Release mode, which generally includes optimizations and excludes debug information.
