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

struct Point {
    int x;
    int y;
};

size_t next_power_of_2(size_t n);

void fft_rec(std::vector<complex> &a, size_t depth = 0);

void fft(std::vector<complex> &a);

void ifft(std::vector<complex> &a);

std::vector<complex> dft(const std::vector<complex> &a);

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
