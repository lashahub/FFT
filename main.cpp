#include "utils.hpp"

using complex = std::complex<double>;


// Function that generates random polynomial coefficients and compares the results of dft and ditfft
bool test(std::vector<complex> X, std::vector<complex> Y) {
    size_t counter = 0;
    bool passed = true;
    for (int i = 0; i < X.size(); ++i) {
        if (std::abs(X[i] - Y[i]) > 0.001) {
            passed = false;
        }
    }
    return passed;
}

// function for testing for M times for given functions
void test_for_M_times(std::vector<complex> (*fun1)(std::vector<complex> &, bool),
                      std::vector<complex> (*fun2)(std::vector<complex> &, size_t), size_t M, size_t N, size_t num_threads) {
    size_t passed = 0;
    for (size_t i = 0; i < M; ++i) {
        std::vector<complex> P(N);
        for (size_t j = 0; j < N; j++) {
            size_t r = rand();
            size_t im = rand();
            P[j] = complex(r % 10, im % 10);
        }
        std::vector<complex> Y = fun2(P, num_threads);
        size_t N_star = modify_size(N, num_threads);
        P.resize(N_star);
        std::vector<complex> X = fun1(P, false);
        X.resize(N);
        bool result = test(X, Y);
        passed += result;
    }
    std::cout << "Passed " << passed << "/" << M << " tests." << std::endl;
}


int main() {
    test_for_M_times(dft, ditfft_parallel, 100, 100, 10);

    return 0;
}

