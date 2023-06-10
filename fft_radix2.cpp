#include "utils.hpp"

complex cmplx(double d);

/**
 * @brief Computes the Fast Fourier Transform (FFT) of a sequence using a radix-2 approach sequentially.
 *
 * This function works by decomposing the sequence into smaller sub-sequences, computes their FFTs,
 * and then combines the results to get the FFT of the original sequence.
 *
 * @param coeff The vector of complex numbers to be transformed.
 */
void fft_radix2_seq(std::vector<complex> &coeff) {
    size_t n = coeff.size();
    if (n == 1)
        return;

    // Separate the input sequence into even and odd indexed parts
    std::vector<complex> even(n / 2), odd(n / 2);

    for (size_t i = 0; 2 * i < n; i++) {
        even[i] = coeff[2 * i];
        if (2 * i + 1 < n) {
            odd[i] = coeff[2 * i + 1];
        }
    }

    // Recursively perform FFT on the even and odd subsequences
    fft_radix2_seq(even);
    fft_radix2_seq(odd);

    // Combine the transformed subsequences to get the FFT of the original sequence
    for (size_t i = 0; 2 * i < n; i++) {
        auto t = cmplx((double) i * 2 * M_PI / (double) n);
        coeff[i] = even[i] + t * odd[i];
        coeff[i + n / 2] = even[i] - t * odd[i];
    }
}


/**
 * @brief Computes the Fast Fourier Transform (FFT) of a sequence using a radix-2 approach in parallel.
 *
 * This function works by decomposing the sequence into smaller sub-sequences, computes their FFTs in parallel,
 * and then combines the results to get the FFT of the original sequence.
 *
 * @param coeff The vector of complex numbers to be transformed.
 * @param num_threads The number of threads to be used for computation.
 */
void fft_radix2_par(std::vector<complex> &coeff, size_t num_threads) {
    // If the number of threads is 1, perform FFT sequentially.
    if (num_threads == 1) {
        fft_radix2_seq(coeff);
        return;
    }

    size_t n = coeff.size();
    if (n == 1)
        return;

    // Initialize a vector of threads.
    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1);

    // Separate the input sequence into even and odd indexed parts
    std::vector<complex> even(n / 2), odd(n / 2);

    // Create threads to separate the sequence
    for (size_t t_i = 1; t_i < num_threads; t_i++) {
        threads.emplace_back([&, t_i] {
            for (size_t i = t_i; 2 * i < n; i += num_threads) {
                even[i] = coeff[2 * i];
                if (2 * i + 1 < n) {
                    odd[i] = coeff[2 * i + 1];
                }
            }
        });
    }
    // Separate the sequence for the first thread
    for (size_t i = 0; 2 * i < n; i += num_threads) {
        even[i] = coeff[2 * i];
        if (2 * i + 1 < n) {
            odd[i] = coeff[2 * i + 1];
        }
    }
    // Wait for all threads to finish
    join_threads(threads);

    // Perform FFT on the even and odd subsequences in parallel
    std::thread t1(fft_radix2_par, std::ref(even), (num_threads + 1) / 2);
    std::thread t2(fft_radix2_par, std::ref(odd), (num_threads + 1) / 2);
    t1.join();
    t2.join();

    // Combine the transformed subsequences to get the FFT of the original sequence
    for (size_t t_i = 1; t_i < num_threads; t_i++) {
        threads.emplace_back([&, t_i] {
            for (size_t i = t_i; 2 * i < n; i += num_threads) {
                auto t = cmplx((double) i * 2 * M_PI / (double) n);
                coeff[i] = even[i] + t * odd[i];
                coeff[i + n / 2] = even[i] - t * odd[i];
            }
        });
    }
    // Combine for the first thread
    for (size_t i = 0; 2 * i < n; i += num_threads) {
        auto t = cmplx((double) i * 2 * M_PI / (double) n);
        coeff[i] = even[i] + t * odd[i];
        coeff[i + n / 2] = even[i] - t * odd[i];
    }
    // Wait for all threads to finish
    join_threads(threads);
}
