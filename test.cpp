#include "test.hpp"

bool isClose(const std::vector<complex> &coeff, size_t N, const std::vector<complex> &test) {
    std::vector<complex> corr = dft_seq(coeff, N);
    for (size_t i = 0; i < corr.size(); ++i) {
        if (std::abs(corr[i] - test[i]) > 0.001) {
            return false;
        }
    }
    return true;
}


bool test_correctness(MODE mode, int num_threads) {
    if (mode == MODE::DFT_SEQ) {
        return true;
    }

    size_t num_tests = 100;
    size_t max_N = 1000;
    size_t max_value = 1000;

    for (size_t i = 0; i < num_tests; i++) {
        size_t N = r_int() % max_N;
        std::vector<complex> coeff(N);
        for (size_t j = 0; j < N; j++) {
            coeff[j] = {(double) (r_int() % max_value), (double) (r_int() % max_value)};
        }
        auto test = coeff;

        if (!isClose(coeff,
                     new_N(mode, num_threads, N),
                     fft(mode, test, num_threads))) {
            return false;
        }
    }
    return true;
}

bool test_all_correctness() {
    std::vector<MODE> modes = {MODE::DFT_SEQ,
                               MODE::DFT_PAR,
                               MODE::FFT_RADIX2_SEQ,
                               MODE::FFT_RADIX2_PAR,
                               MODE::FFT_DIT_INPLACE_SEQ,
                               MODE::FFT_DIT_INPLACE_PAR,
                               MODE::FFT_DIT_SEQ,
                               MODE::FFT_DIT_PAR};
    std::vector<int> num_threads = {1, 2, 4, 8};
    for (auto mode: modes) {
        for (auto num_thread: num_threads) {
            if (!test_correctness(mode, num_thread)) {
                return false;
            }
        }
    }
    return true;
}


double test_performance(MODE mode, size_t num_threads, size_t N) {
    size_t num_tests = 10;
    std::vector<complex> coeff(N);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds{};
    double time = 0;
    for (size_t i = 0; i < num_tests; i++) {
        for (size_t j = 0; j < N; j++) {
            coeff[j] = {(double) (r_int() % 1000), (double) (r_int() % 1000)};
        }
        start = std::chrono::system_clock::now();
        fft(mode, coeff, num_threads);
        end = std::chrono::system_clock::now();
        elapsed_seconds = end - start;
        time += elapsed_seconds.count();
    }
    return time / (double) num_tests;
}

std::vector<std::vector<std::vector<double>>> benchmark(const std::vector<MODE> &modes,
                                                        const std::vector<size_t> &num_threads,
                                                        const std::vector<size_t> &N) {
    std::vector<std::vector<std::vector<double>>> res(modes.size());
    for (size_t i = 0; i < modes.size(); ++i) {
        res[i].resize(num_threads.size());
        for (size_t j = 0; j < num_threads.size(); ++j) {
            res[i][j].resize(N.size());
            for (size_t k = 0; k < N.size(); ++k) {
                res[i][j][k] = test_performance(modes[i], num_threads[j], N[k]);
            }
        }
    }
    return res;
}
