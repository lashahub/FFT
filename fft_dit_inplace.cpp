#include "utils.hpp"

// Function to perform the DIT-FFT algorithm sequentially and in-place
void fft_dit_inplace_seq(std::vector<complex> &P, size_t start, size_t stride, size_t N) {
    if (N <= 1) {
        return;
    }

    // Get a prime factor of N
    size_t p = get_prime_factor(N);
    size_t q = N / p;

    if (p == N) {
        // Perform the FFT directly if N is prime
        if (p == 2) {
            // Base case for N = 2
            complex t = P[start];
            P[start] = t + P[start + stride];
            P[start + stride] = t - P[start + stride];
            return;
        }
        // Otherwise, use the DFT algorithm for prime N
        std::vector<complex> temp(N);
        for (int i = 0; i < N; i++) {
            temp[i] = P[start + i * stride];
        }
        temp = dft_seq(temp);
        for (int i = 0; i < N; i++) {
            P[start + i * stride] = temp[i];
        }
        return;
    }

    // Perform the DIT-FFT recursively
    for (int i = 0; i < p; i++) {
        fft_dit_inplace_seq(P, start + i * stride, stride * p, q);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P[start + stride * (j * p + i)] *= cmplx(2 * M_PI * i * j / (double) N);
        }
    }

    // Recombine the results
    for (int i = 0; i < q; i++) {
        fft_dit_inplace_seq(P, start + stride * i * p, stride, p);
    }

    std::vector<complex> temp(p * q);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            temp[i * q + j] = P[start + stride * (j * p + i)];
        }
    }

    for (int i = 0; i < p * q; i++) {
        P[start + stride * i] = temp[i];
    }
}

// Function to perform the DIT-FFT algorithm in parallel
void fft_dit_inplace_par(std::vector<complex> &P, size_t start, size_t stride, size_t N, size_t num_threads) {
    if (num_threads == 1) {
        // If only one thread, perform the sequential FFT
        fft_dit_inplace_seq(P, start, stride, N);
        return;
    }

    if (N <= 1) {
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1);

    // Get a prime factor of N
    size_t p = get_prime_factor(N);
    size_t q = N / p;

    if (p == N) {
        // Perform the FFT directly if N is prime
        std::vector<complex> temp(N);
        for (size_t i = 0; i < N; i++) {
            temp[i] = P[start + i * stride];
        }
        temp = dft_par(temp, num_threads);
        for (size_t i = 0; i < N; i++) {
            P[start + i * stride] = temp[i];
        }
        return;
    }

    if (p > num_threads) {
        // If the number of prime factors is greater than the number of threads, divide the work among the threads
        size_t tasks_per_thread = (p + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back([&, i] {
                for (size_t j = i * tasks_per_thread; j < std::min((i + 1) * tasks_per_thread, p); j++) {
                    fft_dit_inplace_seq(P, start + j * stride, stride * p, q);
                }
            });
        }
        join_threads(threads);
    } else {
        // If the number of threads is greater than or equal to the number of prime factors, divide the threads equally among the factors
        size_t threads_per_task = (num_threads + p - 1) / p;
        for (size_t i = 0; i < p; i++) {
            threads.emplace_back([&, i] {
                fft_dit_inplace_par(P, start + i * stride, stride * p, q, threads_per_task);
            });
        }
        join_threads(threads);
    }

    for (size_t t = 1; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            for (size_t index = t; index < N; index += num_threads) {
                size_t i = index / q;
                size_t j = index % q;
                P[start + stride * (j * p + i)] *= cmplx(2 * M_PI * (double) i * (double) j / (double) N);
            }
        });
    }
    for (size_t index = 0; index < N; index += num_threads) {
        size_t i = index / q;
        size_t j = index % q;
        P[start + stride * (j * p + i)] *= cmplx(2 * M_PI * (double) i * (double) j / (double) N);
    }
    join_threads(threads);

    if (q > num_threads) {
        // If the number of subproblems is greater than the number of threads, divide the work among the threads
        size_t tasks_per_thread = (q + num_threads - 1) / num_threads;
        for (size_t i = 0; i < num_threads; i++) {
            threads.emplace_back([&, i] {
                for (size_t j = i * tasks_per_thread; j < std::min((i + 1) * tasks_per_thread, q); j++) {
                    fft_dit_inplace_seq(P, start + stride * j * p, stride, p);
                }
            });
        }
        join_threads(threads);
    } else {
        // If the number of threads is greater than or equal to the number of subproblems, divide the threads equally among the subproblems
        size_t threads_per_task = (num_threads + q - 1) / q;
        for (size_t i = 0; i < q; i++) {
            threads.emplace_back([&, i] {
                fft_dit_inplace_par(P, start + stride * i * p, stride, p, threads_per_task);
            });
        }
        join_threads(threads);
    }

    std::vector<complex> temp(p * q);

    for (size_t t = 1; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            for (size_t index = t; index < N; index += num_threads) {
                size_t i = index / q;
                size_t j = index % q;
                temp[index] = P[start + stride * (j * p + i)];
            }
        });
    }
    for (size_t index = 0; index < N; index += num_threads) {
        size_t i = index / q;
        size_t j = index % q;
        temp[index] = P[start + stride * (j * p + i)];
    }
    join_threads(threads);

    // Assign the gathered results to the correct positions in the output array
    for (size_t t = 1; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            for (size_t i = t; i < N; i += num_threads) {
                P[start + stride * i] = temp[i];
            }
        });
    }
    for (size_t i = 0; i < N; i += num_threads) {
        P[start + stride * i] = temp[i];
    }
    join_threads(threads);
}
