#include "utils.hpp"


// This function computes the Fast Fourier Transform (FFT) of a sequence in a sequential manner.
std::vector<complex> fft_dit_seq(std::vector<complex> &P) {
    size_t N = P.size(); // Size of input sequence.

    // Base case for recursion.
    if (N == 1) {
        return P;
    }

    // Get a prime factor of the size of the sequence.
    size_t p = get_prime_factor(N);

    // If the size is a prime number, perform DFT sequentially.
    if (p == N) {
        return dft_seq(P);
    }

    // Decompose the sequence into sub-sequences.
    size_t q = N / p;
    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            A[i][j] = P[j * p + i];
        }
    }

    // Perform FFT on each sub-sequence.
    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (int i = 0; i < p; ++i) {
        B[i] = fft_dit_seq(A[i]);
    }

    // Multiply by the twiddle factors.
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            B[i][j] *= cmplx(2 * M_PI * i * j / (double) N);
        }
    }

    // Transpose the matrix.
    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = B[j][i];
        }
    }

    // Perform FFT on each sub-sequence.
    for (int i = 0; i < q; i++) {
        C[i] = fft_dit_seq(C[i]);
    }

    // Construct the final sequence.
    std::vector<complex> P_star(N);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P_star[i * q + j] = C[j][i];
        }
    }

    return P_star;
}

/**
 * @brief Computes the Fast Fourier Transform (FFT) of a sequence using a Decimation in Time (DIT) approach.
 *
 * This function performs the FFT in parallel manner, distributing the computation over multiple threads.
 * It works by decomposing the original sequence into smaller sub-sequences, computes their FFTs,
 * and then combines the results to get the FFT of the original sequence. It uses an in-place computation of the FFT on each subsequence.
 *
 * @param P The vector of complex numbers to be transformed.
 * @param start The starting position of the sequence to be transformed.
 * @param stride The stride between the elements of the sequence.
 * @param N The size of the sequence to be transformed.
 * @param num_threads The number of threads to be used for computation.
 */
void fft_dit_par(std::vector<complex> &P, size_t start, size_t stride, size_t N, size_t num_threads) {
    // If the number of threads is 1, perform FFT sequentially.
    if (num_threads == 1) {
        fft_dit_inplace_seq(P, start, stride, N);
        return;
    }

    // Base case for recursion.
    if (N == 1) {
        return;
    }

    // Decompose the sequence into sub-sequences.
    size_t p = num_threads;
    size_t q = N / p;

    // Initialize a vector of threads.
    std::vector<std::thread> threads;
    threads.reserve(num_threads - 1);

    // Determine how many tasks each thread will perform.
    size_t tasks_per_thread = (p + num_threads - 1) / num_threads;

    // Create threads and perform FFT on each sub-sequence.
    for (size_t i = 0; i < num_threads; i++) {
        threads.emplace_back([&, i] {
            for (size_t j = i * tasks_per_thread; j < std::min((i + 1) * tasks_per_thread, p); j++) {
                fft_dit_inplace_seq(P, start + j * stride, stride * p, q);
            }
        });
    }
    // Wait for all threads to finish.
    join_threads(threads);

    // Apply the twiddle factor on each sub-sequence.
    for (size_t t = 1; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            for (size_t index = t; index < N; index += num_threads) {
                size_t i = index / q;
                size_t j = index % q;
                P[start + stride * (j * p + i)] *= cmplx(2 * M_PI * (double) i * (double) j / (double) N);
            }
        });
    }

    // Repeat for the first thread.
    for (size_t index = 0; index < N; index += num_threads) {
        size_t i = index / q;
        size_t j = index % q;
        P[start + stride * (j * p + i)] *= cmplx(2 * M_PI * (double) i * (double) j / (double) N);
    }
    // Wait for all threads to finish.
    join_threads(threads);

    // Determine how many tasks each thread will perform for the next set of operations.
    tasks_per_thread = (q + num_threads - 1) / num_threads;

    // Perform FFT on the transformed sub-sequences.
    for (size_t i = 0; i < num_threads; i++) {
        threads.emplace_back([&, i] {
            for (size_t j = i * tasks_per_thread; j < std::min((i + 1) * tasks_per_thread, q); j++) {
                fft_dit_inplace_seq(P, start + stride * j * p, stride, p);
            }
        });
    }
    // Wait for all threads to finish.
    join_threads(threads);

    // Transpose the matrix formed by the sub-sequences and store in temp.
    std::vector<complex> temp(p * q);

    // Assign the transposed sequences back to the original vector P.
    for (size_t t = 1; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            for (size_t index = t; index < N; index += num_threads) {
                size_t i = index / q;
                size_t j = index % q;
                temp[index] = P[start + stride * (j * p + i)];
            }
        });
    }

    // Repeat for the first thread.
    for (size_t index = 0; index < N; index += num_threads) {
        size_t i = index / q;
        size_t j = index % q;
        temp[index] = P[start + stride * (j * p + i)];
    }
    // Wait for all threads to finish.
    join_threads(threads);

    // Assign the values in temp back to the original vector P.
    for (size_t t = 1; t < num_threads; t++) {
        threads.emplace_back([&, t] {
            for (size_t i = t; i < N; i += num_threads) {
                P[start + stride * i] = temp[i];
            }
        });
    }

    // Repeat for the first thread.
    for (size_t i = 0; i < N; i += num_threads) {
        P[start + stride * i] = temp[i];
    }
    // Wait for all threads to finish.
    join_threads(threads);
}
