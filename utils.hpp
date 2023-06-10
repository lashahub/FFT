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

using complex = std::complex<double>;

extern std::random_device rd;
extern std::uniform_int_distribution<> uniform;
extern std::mt19937_64 engine;

enum class MODE {
    DFT_SEQ,
    DFT_PAR,
    FFT_RADIX2_SEQ,
    FFT_RADIX2_PAR,
    FFT_DIT_INPLACE_SEQ,
    FFT_DIT_INPLACE_PAR,
    FFT_DIT_SEQ,
    FFT_DIT_PAR
};

// Auxiliary functions --------------------------------------------------------

size_t next_power_of_2(size_t n);

size_t get_prime_factor(size_t N);

size_t modify_size(size_t N, size_t p);

complex cmplx(double angle);

void join_threads(std::vector<std::thread> &threads);

size_t new_N(MODE mode, size_t num_threads, size_t N);

size_t r_int();

// Main FFT -------------------------------------------------------------------

std::vector<complex>
fft(MODE mode, std::vector<complex> &coeff, size_t num_threads = 1);

// DFT ------------------------------------------------------------------------

std::vector<complex> dft_seq(const std::vector<complex> &coeff, size_t N = 0);

std::vector<complex> dft_par(const std::vector<complex> &coeff, size_t num_threads, size_t N = 0);

// FFT ------------------------------------------------------------------------

void fft_radix2_seq(std::vector<complex> &coeff);

void fft_radix2_par(std::vector<complex> &coeff, size_t num_threads);

void fft_dit_inplace_seq(std::vector<complex> &P, size_t start, size_t stride, size_t N);

void fft_dit_inplace_par(std::vector<complex> &P, size_t start, size_t stride, size_t N, size_t num_threads);

std::vector<complex> fft_dit_seq(std::vector<complex> &P);

void fft_dit_par(std::vector<complex> &P, size_t start, size_t stride, size_t N, size_t num_threads);

// Image processing functions -------------------------------------------------

struct Point {
    int x;
    int y;
};

Point findStartingPoint(const uint8_t *image, int width, int height, int channels);

std::vector<Point> traceContour(const uint8_t *image, int width, int height, int channels, Point start);

std::vector<Point> transformContour(const std::vector<Point> &contour, int numFrequencies);

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
