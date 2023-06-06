
#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <thread>
#include <algorithm>
#include <random>

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

//#include "utils.hpp"

const size_t MAX_THREAD_DEPTH = 4;
using complex = std::complex<double>;

std::random_device rd;
std::uniform_int_distribution<> uniform(0, std::numeric_limits<int>::max());
std::mt19937_64 engine(rd());

// helper function to join threads given thread vector
void join_threads(std::vector<std::thread> &threads) {
    for (auto &thread : threads) {
        thread.join();
    }
    threads.clear();
}

// DFT function
std::vector<complex> dft(std::vector<complex> &x) {
    std::vector<complex> X(x.size());
    for (int k = 0; k < x.size(); ++k) {
        for (int n = 0; n < x.size(); ++n) {
            X[k] += x[n] * std::exp(complex(0, -2 * M_PI * k * n / (double) x.size()));
        }
    }
    return X;
}

// FFT function radix2-cooley-tukey
std::vector<complex> fft(const std::vector<complex> &x) {
    std::vector<complex> X(x.size());
    int N = x.size();

    if (x.size() == 1) {
        X[0] = x[0];
        return X;
    }
    std::vector<complex> x_even(x.size() / 2);
    std::vector<complex> x_odd(x.size() / 2);
    for (int i = 0; i < x.size() / 2; ++i) {
        x_even[i] = x[2 * i];
        x_odd[i] = x[2 * i + 1];
    }
    std::vector<complex> X_even = fft(x_even);
    std::vector<complex> X_odd = fft(x_odd);

    complex w = std::exp(complex(0, 0));
    complex w_n = std::exp(complex(0, -2 * M_PI / N));
    for (int k = 0; k < x.size() / 2; k++) {
        X[k] = X_even[k] + w * X_odd[k];
        X[k + x.size() / 2] = X_even[k] - w * X_odd[k];
        w = w * w_n;
    }
    return X;
}

// get prime factors of N
int get_prime_factors(int N) {
    std::vector<int> factors;
    int p = 2;
    while (N > 1) {
        if (N % p == 0) {
            break;
        }
        p++;
    }
    return p;
}

//DITFFT function which uses fft function
std::vector<complex> ditfft(std::vector<complex> &P) {
    int N = P.size();
    int p = get_prime_factors(N);

    if (p == N) {
        return dft(P);
    }

    int q = N / p;
    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            A[i][j] = P[j * p + i];
        }
    }

    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (int i = 0; i < p; ++i) {
        B[i] = ditfft(A[i]);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            B[i][j] *= std::exp(complex(0, -2 * M_PI * i * j / N));
        }
    }

    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = B[j][i];
        }
    }

    for (int i = 0; i < q; i++) {
        C[i] = ditfft(C[i]);
    }

    std::vector<complex> P_star(N);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P_star[i * q + j] = C[j][i];
        }
    }

    return P_star;
}

//adjust size
std::vector<complex> adjust_size(std::vector<complex> &P, int new_size) {
    std::vector<complex> P_new(new_size);
    for (int i = 0; i < P.size(); i++) {
        P_new[i] = P[i];
    }
    return P_new;
}


//parallel ditfft
std::vector<complex> parallel_ditfft(std::vector<complex> &P_OLD, int num_threads) {
    std::vector<complex> P(P_OLD.size());
    if (P_OLD.size() % num_threads != 0){
        P = adjust_size(P_OLD, P_OLD.size() + num_threads - (P_OLD.size() % num_threads));
    } else {
        P = adjust_size(P_OLD, P_OLD.size());
    }

    std::cout << P_OLD.size() << " " << P.size() << std::endl;

    int N = P.size();
    int p = num_threads;
    int q = N / p;

    std::vector<std::thread> threads;

    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (int i = 0; i < p - 1; i++) {
        threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < q; j++) {
                A[i][j] = P[j * p + i];
            }
        }));
    }
    for (int j = 0; j < q; j++) {
        A[p - 1][j] = P[j * p + p - 1];
    }

    join_threads(threads);

    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (int i = 0; i < p - 1; i++) {
         threads.push_back(std::thread([&, i]() {
            B[i] = ditfft(A[i]);
         }));
    }
    B[p - 1] = ditfft(A[p - 1]);

    join_threads(threads);

    for (int i = 0; i < p-1; i++) {
         threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < q; j++) {
                B[i][j] *= std::exp(complex(0, -2 * M_PI * i * j / N));
            }
         }));
    }
    for (int j = 0; j < q; j++) {
        B[p - 1][j] *= std::exp(complex(0, -2 * M_PI * (p - 1) * j / N));
    }

    join_threads(threads);

    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (int i = 0; i < q; i++) {
         threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < p; j++) {
                C[i][j] = B[j][i];
            }
         }));
    }

    join_threads(threads);

    for (int i = 0; i < q; i++) {
        C[i] = ditfft(C[i]);
    }

    std::vector<complex> P_star(N);
    for (int i = 0; i < p - 1; i++) {
        threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < q; j++) {
                P_star[i * q + j] = C[j][i];
            }
        }));
    }
    for (int j = 0; j < q; j++) {
        P_star[(p - 1) * q + j] = C[j][p - 1];
    }

    join_threads(threads);

    return P_star;
}


/*void fft(std::vector<std::complex<double>>& a, int start, int step, int size) {
    if (size < 2) {
        return;
    }

    // compute half-size FFTs
    fft(a, start, 2*step, size/2);
    fft(a, start+step, 2*step, size/2);

    // combine results
    for (int i = 0; i < size/2; i++) {
        std::complex<double> t = std::exp(std::complex<double>(0, -2.0 * PI * i / size)) * a[start + 2*step*i + step];
        a[start + 2*step*i + step] = a[start + 2*step*i] - t;
        a[start + 2*step*i] += t;
    }
}

void mixedRadixFft(std::vector<std::complex<double>>& a, std::vector<int>& factors) {
    int N = a.size();
    int product = 1;

    for (int factor : factors) {
        int step = N / (product * factor);
        for (int i = 0; i < product; i++) {
            fft(a, step*i, step, factor);
        }
        product *= factor;
    }
}*/

// Function that generates random polynomial coefficients and compares the results of dft and ditfft
std::pair<bool, int> test(std::vector<complex> (*fun1)(std::vector<complex> &),
                          std::vector<complex> (*fun2)(std::vector<complex> &), int N) {
    std::vector<complex> P(N);
    for (int i = 0; i < N; ++i) {
        int r = rand();
        int im = rand();
        P[i] = complex(r % 10, im % 10);
    }
    bool passed = true;
    int counter = 0;
    std::vector<complex> X = fun1(P);
    std::vector<complex> Y = fun2(P);
    for (int i = 0; i < X.size(); ++i) {
        if (std::abs(X[i] - Y[i]) > 0.001) {
            counter++;
            std::cout << "Failed!" << std::endl;
            passed = false;
        }
    }
    return {passed, counter};
}

// function for testing for M times for given functions
void test_for_M_times(std::vector<complex> (*fun1)(std::vector<complex> &),
                      std::vector<complex> (*fun2)(std::vector<complex> &), int M, int N) {
    int passed = 0;
    for (int i = 0; i < M; ++i) {
        std::pair<bool, int> result = test(fun1, fun2, N);
        passed += result.first;
    }
    std::cout << "Passed " << passed << "/" << M << " tests" << std::endl;
}


int main() {

    //test_for_M_times(dft, ditfft, 100, 512);
    //test_for_M_times(dft, fft, 100, 512);

    int N = 3;
    std::vector<complex> P(N);
    for (int i = 0; i < N; i++) {
        // generate a random integer
        int r = i;
        int im = 0;
        P[i] = complex(r, im);
    }

//    std::vector<complex> X = parallel_ditfft(P, 5);
    P.push_back(complex(0, 0));
//    P.push_back(complex(0, 0));
//    P.push_back(complex(0, 0));
//    P.push_back(complex(0, 0));

    std::vector<complex> Y = dft(P);
    for (int i = 0; i < Y.size(); i++) {
        std::cout << Y[i] << std::endl;
    }

    return 0;
}

