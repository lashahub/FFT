#pragma once

#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <thread>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>

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

// FFT functions

void radix2_fft_parallel(std::vector<complex> &coeff, std::vector<complex> &eval, size_t depth = 0);

void radix2_fft_sequential(std::vector<complex> &coeff, std::vector<complex> &eval);

std::vector<complex> fft(std::vector<complex> &coeff);

std::vector<complex> ifft(std::vector<complex> &coeff);

std::vector<complex> dft(std::vector<complex> &a, bool power_of_2 = false);

std::vector<complex> ditfft_sequential(std::vector<complex> &P);

void ditfft2_inplace(std::vector<complex>& P, size_t start, size_t stride, size_t N);

void radix2_fft_parallel(std::vector<complex> &coeff, std::vector<complex> &eval, size_t depth);

std::vector<complex> ditfft_parallel(std::vector<complex> &P, size_t num_threads);

// Image processing functions

Point findStartingPoint(const uint8_t *image, int width, int height, int channels);

std::vector<Point> traceContour(const uint8_t *image, int width, int height, int channels, Point start);

std::vector<Point> transformContour(const std::vector<Point> &contour, int numFrequencies);

void
printDesmosEquations(const std::vector<complex> &contour_x, const std::vector<complex> &contour_y, int numFrequencies);

/*
 * FFT implementation from https://cp-algorithms.com/algebra/fft.html
 */

const int64_t mod = (119 << 23) + 1, root = 3;  // mod is a large prime number, root is primitive root modulo mod
const int MAXN = (1 << 20);

int64_t power(int64_t a, int64_t b, int64_t p);

void fft(std::vector<int64_t> &a, bool invert);

std::vector<int64_t> multiply(std::vector<int64_t> const &a, std::vector<int64_t> const &b);

/*
 * Audio processing
 */

struct WavHeader {
    char chunkId[4];
    uint32_t chunkSize;
    char format[4];
    char subchunk1Id[4];
    uint32_t subchunk1Size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char subchunk2Id[4];
    uint32_t subchunk2Size;
};

struct WavData {
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint16_t bitsPerSample;
    std::vector<uint16_t> leftChannel;
    std::vector<uint16_t> rightChannel;
};

WavData readWavFile(const std::string &filename);
