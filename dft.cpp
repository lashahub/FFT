#include "utils.hpp"

// This function computes the Discrete Fourier Transform (DFT) of a sequence in a sequential manner.
std::vector<complex> dft_seq(const std::vector<complex> &coeff, size_t N) {
    N = (N == 0) ? coeff.size() : N; // If N is not specified, set it to coeff.size().
    std::vector<complex> eval(coeff.size()); // Initialize the vector that will hold the DFT of the sequence.

    // Compute the DFT of the sequence.
    for (size_t k = 0; k < coeff.size(); k++) {
        for (size_t j = 0; j < coeff.size(); j++) {
            eval[k] += coeff[j] * cmplx(2 * M_PI * (double) j * (double) k / (double) N);
        }
    }
    return eval;
}

// This function computes the Discrete Fourier Transform (DFT) of a sequence in a parallel manner.
std::vector<complex> dft_par(const std::vector<complex> &coeff, size_t num_threads, size_t N) {
    N = (N == 0) ? coeff.size() : N; // If N is not specified, set it to coeff.size().

    // If the number of threads is 1, then compute the DFT of the sequence in a sequential manner.
    if (num_threads == 1) {
        return dft_seq(coeff, N);
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1);

    std::vector<complex> eval(coeff.size()); // Initialize the vector that will hold the DFT of the sequence.

    // Compute the DFT of the sequence in a parallel manner.
    for (size_t i = 1; i < num_threads; i++) {
        threads.emplace_back([i, &coeff, &eval, N, num_threads]() {
            // Each thread calculates a subset of the DFT.
            for (size_t k = i; k < coeff.size(); k += num_threads) {
                for (size_t j = 0; j < coeff.size(); j++) {
                    eval[k] += coeff[j] * cmplx(2 * M_PI * (double) j * (double) k / (double) N);
                }
            }
        });
    }

    // The main thread calculates the remaining subset of the DFT.
    for (size_t k = 0; k < coeff.size(); k += num_threads) {
        for (size_t j = 0; j < coeff.size(); j++) {
            eval[k] += coeff[j] * cmplx(2 * M_PI * (double) j * (double) k / (double) N);
        }
    }
    join_threads(threads);

    return eval;
}
