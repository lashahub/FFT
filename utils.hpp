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

std::vector<Point> transformContour(const std::vector<Point> &contour);
