#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <thread>
#include <algorithm>
#include <random>

const int MAX_THREAD_DEPTH = 4;
using complex = std::complex<double>;

std::random_device rd;
std::uniform_int_distribution<> uniform(0, std::numeric_limits<int>::max());
std::mt19937_64 engine(rd());

void fft(std::vector<complex> &a, size_t n, int depth = 0) {
    size_t length = a.size();
    if (length == 1)
        return;

    std::vector<complex> a0((length + 1) / 2), a1(length / 2);
    for (int i = 0; 2 * i < length; i++) {
        a0[i] = a[2 * i];
        if (2 * i + 1 < length) {
            a1[i] = a[2 * i + 1];
        }
    }

    if (depth < MAX_THREAD_DEPTH && false) {
        std::thread t1(fft, std::ref(a0), n / 2, depth + 1);
        std::thread t2(fft, std::ref(a1), n / 2, depth + 1);
        t1.join();
        t2.join();
    } else {
        fft(a0, depth + 1);
        fft(a1, depth + 1);
    }

    double ang = -2 * M_PI / (double) (n);
    complex w(1), w_n(std::exp(complex(0, ang)));

    for (int i = 0; i < length / 2; i++) {
        a[i] = a0[i] + w * a1[i];
        if (i + length / 2 < length) {
            a[i + length / 2] = a0[i] - w * a1[i];
        }
        w = w * w_n;
    }
}

std::vector<complex> dft(const std::vector<complex> &a, size_t n) {
    size_t length = a.size();
    std::vector<complex> y(length);
    for (size_t k = 0; k < length; k++) {
        for (size_t j = 0; j < length; j++) {
            double angle = 2 * M_PI * (double) j * (double) k / (double) n;
            y[k] += a[j] * complex(std::cos(angle), -std::sin(angle));
        }
    }
    return y;
}

bool compareResults(const std::vector<complex> &fft_res, const std::vector<complex> &dft_res) {
    double epsilon = 1e-6;
    if (fft_res.size() != dft_res.size()) {
        return false;
    }
    for (size_t i = 0; i < fft_res.size(); i++) {
        if (std::abs(fft_res[i].real() - dft_res[i].real()) > epsilon ||
            std::abs(fft_res[i].imag() - dft_res[i].imag()) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    size_t length = 3, n = 4;

    std::vector<complex> a(length);
    for (int i = 0; i < length; i++) {
        a[i] = complex(i, 0);
//        a[i] = complex(uniform(engine) % 1000, 0);
    }

    auto b = dft(a, n);
    fft(a, n);

    // print a and b
    for (int i = 0; i < length; i++) {
        std::cout << a[i] << " " << b[i] << std::endl;
    }

    if (compareResults(a, b)) {
        std::cout << "FFT and DFT results match" << std::endl;
    } else {
        std::cout << "FFT and DFT results do not match" << std::endl;
    }

    return 0;
}
