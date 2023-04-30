#include <iostream>
#include <complex>
#include <vector>

using complex = std::complex<double>;

std::vector<complex> dft(const std::vector<complex> &x) {
    std::vector<complex> X(x.size());
    for (int k = 0; k < x.size(); ++k) {
        for (int n = 0; n < x.size(); ++n) {
            X[k] += x[n] * std::exp(complex(0, -2 * M_PI * k * n / (double) x.size()));
        }
    }
    return X;
}

int main() {
    // create a vector of 440hz sine wave samples, sampled at 44100hz
    std::vector<complex> x(6000);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = std::sin(2 * M_PI * 440 * (double) i / 44100.0);
    }
    // perform dft
    std::vector<complex> X = dft(x);
    // print the first 10 values
    for (size_t i = 0; i < 500; ++i) {
        std::cout << i << " " << std::norm(X[i]) << std::endl;
    }
    return 0;
}
