#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <thread>
#include <algorithm>
#include <random>
#include <iomanip>

const int MAX_THREAD_DEPTH = 4;
using complex = std::complex<double>;

std::random_device rd;
std::uniform_int_distribution<> uniform(0, std::numeric_limits<int>::max());
std::mt19937_64 engine(rd());

size_t next_power_of_2(size_t n) {
    size_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}

void fft_rec(std::vector<complex> &a, int depth = 0) {
    size_t n = a.size();
    if (n == 1)
        return;

    std::vector<complex> even(n / 2), odd(n / 2);
    for (int i = 0; 2 * i < n; i++) {
        even[i] = a[2 * i];
        if (2 * i + 1 < n) {
            odd[i] = a[2 * i + 1];
        }
    }

    if (depth < MAX_THREAD_DEPTH) {
        std::thread t1(fft_rec, std::ref(even), depth + 1);
        std::thread t2(fft_rec, std::ref(odd), depth + 1);
        t1.join();
        t2.join();
    } else {
        fft_rec(even, depth + 1);
        fft_rec(odd, depth + 1);
    }

    double ang = 2 * M_PI / (double) n;
    complex w(1), wn(std::cos(ang), std::sin(ang));

    for (size_t i = 0; 2 * i < n; i++) {
        a[i] = even[i] + w * odd[i];
        a[i + n / 2] = even[i] - w * odd[i];
        w *= wn;
    }
}

void fft(std::vector<complex> &a) {
    size_t n = a.size();
    a.resize(next_power_of_2(n));
    fft_rec(a);
    a.resize(n);
}

std::vector<complex> dft(const std::vector<complex> &a) {
    size_t length = a.size();
    size_t n = next_power_of_2(length);
    std::vector<complex> y(length);
#pragma omp parallel for
    for (size_t k = 0; k < length; k++) {
        for (size_t j = 0; j < length; j++) {
            double angle = 2 * M_PI * (double) j * (double) k / (double) n;
            y[k] += a[j] * complex(std::cos(angle), std::sin(angle));
        }
    }
    return y;
}

bool compareResults(const std::vector<complex> &fft_res, const std::vector<complex> &dft_res) {
    double epsilon = 1e-2;
    if (fft_res.size() != dft_res.size()) {
        return false;
    }
    for (size_t i = 0; i < fft_res.size(); i++) {
        if (std::abs(fft_res[i] - dft_res[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    size_t n = 3911;

    std::vector<complex> a(n);
    for (int i = 0; i < n; i++) {
        a[i] = complex(uniform(engine) % 100, 0);
    }

    auto b = dft(a);
    fft(a);


    if (compareResults(a, b)) {
        std::cout << "FFT and DFT results match" << std::endl;
    } else {
        std::cout << "FFT and DFT results do not match" << std::endl;
    }

    return 0;
}
