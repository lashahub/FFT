#include <iostream>
#include <complex>
#include <cmath>
#include <vector>

using complex = std::complex<double>;

class Wave {
public:
    double frequency;
    double amplitude;
    double phase;

public:
    Wave(double frequency, double amplitude, double phase)
            : frequency(frequency), amplitude(amplitude),
              phase(phase) {}

};

std::vector<complex> WavesToSamples(const std::vector<Wave> &waves, int sample_rate, double duration) {
    std::vector<complex> samples(duration * sample_rate);
    for (size_t i = 0; i < samples.size(); ++i) {
        for (const auto &wave: waves) {
            samples[i] += wave.amplitude * std::sin(2 * M_PI * wave.frequency * (double) i / (double) sample_rate +
                                                    wave.phase);
        }
    }
    return samples;
}


// DFT function
std::vector<complex> dft(const std::vector<complex> &x) {
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
    for (int k = 0; k < x.size() / 2; ++k) {
        X[k] = X_even[k] + w * X_odd[k];
        X[k + x.size() / 2] = X_even[k] - w * X_odd[k];
        w = w * w_n;
    }
    return X;
}


//FFT General version for composite N = pq
std::vector<complex> fft_general(const std::vector<complex> &x, int N, int p, int q) {
    std::vector<complex> X(x.size());
}


int main() {

    std::vector<Wave> waves = {
            {440,  1,    0},
            {880,  0.5,  0},
            {1320, 0.25, 0}
    };

    std::vector<complex> X = dft(WavesToSamples(waves, 4410, 0.2322));
    std::vector<complex> X2 = fft(WavesToSamples(waves, 4410, 0.2322));
    std::cout << X.size() << std::endl;
    for (size_t i = 0; i < 500; ++i) {
        std::cout << i << " " << std::norm(X2[i] - X[i]) << std::endl;
    }
    std:: cout << X.size() << std::endl;
    return 0;
}

