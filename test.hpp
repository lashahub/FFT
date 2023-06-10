#pragma once

#include "utils.hpp"

bool isClose(const std::vector<complex> &coeff, size_t N, const std::vector<complex> &test);

bool test_correctness(MODE mode, int num_threads);

bool test_all_correctness();

double test_performance(MODE mode, size_t num_threads, size_t N);

std::vector<std::vector<std::vector<double>>> benchmark(const std::vector<MODE> &modes,
                                                      const std::vector<size_t> &num_threads,
                                                      const std::vector<size_t> &N);
