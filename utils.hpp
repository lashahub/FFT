#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <thread>
#include <algorithm>
#include <random>

const size_t MAX_THREAD_DEPTH = 4;
using complex = std::complex<double>;

extern std::random_device rd;
extern std::uniform_int_distribution<> uniform;
extern std::mt19937_64 engine;

//#define STB_IMAGE_IMPLEMENTATION
//
//#include "stb_image.h"
//
//#define STB_IMAGE_WRITE_IMPLEMENTATION
//
//#include "stb_image_write.h"

struct Point {
    int x;
    int y;
};


// Auxilarry functions

size_t next_power_of_2(size_t n);

std::vector<complex> adjust_size(std::vector<complex> &P, int new_size);

size_t get_prime_factor(size_t N);

size_t modify_size(size_t N, size_t p);

void join_threads(std::vector<std::thread> &threads);

complex cmplx(double real);

// FFT functions

void radix2_fft_parallel(std::vector<complex> &coeff, std::vector<complex> &eval, size_t depth = 0);

void radix2_fft_sequential(std::vector<complex> &coeff, std::vector<complex> &eval);

std::vector<complex> fft(std::vector<complex> &coeff);

std::vector<complex> ifft(std::vector<complex> &coeff);

std::vector<complex> dft_seq(std::vector<complex> &a, bool power_of_2 = false);

std::vector<complex> ditfft_sequential(std::vector<complex> &P);

void fft_dit_inplace_seq(std::vector<complex>& P, size_t start, size_t stride, size_t N);

void radix2_fft_parallel(std::vector<complex> &coeff, std::vector<complex> &eval, size_t depth);

std::vector<complex> ditfft_parallel(std::vector<complex> &P, size_t num_threads);

void fft_radix2_seq(std::vector<complex> &coeff, size_t start, size_t stride, size_t n);

std::vector<complex> fft_dit_seq(std::vector<complex> &P);

void fft_dit_par(std::vector<complex> &P, size_t start, size_t stride, size_t N, size_t num_threads);

// Image processing functions

Point findStartingPoint(const uint8_t *image, int width, int height, int channels);

std::vector<Point> traceContour(const uint8_t *image, int width, int height, int channels, Point start);

std::vector<Point> transformContour(const std::vector<Point> &contour);
