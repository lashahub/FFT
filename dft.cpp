#include "utils.hpp"

std::vector<complex> dft_seq(std::vector<complex> &coeff, bool power_of_2) {
    size_t N = coeff.size();
    if (power_of_2) {
        N = next_power_of_2(N);
    }
    std::vector<complex> eval(coeff.size());
    for (size_t k = 0; k < coeff.size(); k++) {
        for (size_t j = 0; j < coeff.size(); j++) {
            eval[k] += coeff[j] * cmplx(2 * M_PI * (double) j * (double) k / (double) N);
        }
    }
    return eval;
}

std::vector<complex> dft_par(std::vector<complex> &coeff, size_t THREADS_AVAILABLE, bool power_of_2) {
    size_t N = coeff.size();
    if (power_of_2) {
        N = next_power_of_2(N);
    }

    std::vector<std::thread> threads;
    threads.reserve(THREADS_AVAILABLE - 1);

    std::vector<complex> eval(coeff.size());
    for (size_t i = 1; i < THREADS_AVAILABLE; i++) {
        threads.emplace_back([i, &coeff, &eval, N, THREADS_AVAILABLE]() {
            for (size_t k = i; k < coeff.size(); k += THREADS_AVAILABLE) {
                for (size_t j = 0; j < coeff.size(); j++) {
                    eval[k] += coeff[j] * cmplx(2 * M_PI * (double) j * (double) k / (double) N);
                }
            }
        });
    }
    for (size_t k = 0; k < coeff.size(); k += THREADS_AVAILABLE) {
        for (size_t j = 0; j < coeff.size(); j++) {
            eval[k] += coeff[j] * cmplx(2 * M_PI * (double) j * (double) k / (double) N);
        }
    }
    join_threads(threads);

    return eval;
}
