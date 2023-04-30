#include <iostream>
#include <complex>
#include <vector>
#include <random>

#define NUM_SAMPLES 4096

static std::random_device rd;
using rand_eng = std::mt19937_64;
static rand_eng eng(rd());
static std::uniform_real_distribution<> uniform_real(0.0, 1.0);
static std::uniform_int_distribution<> uniform_int(0);

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

std::vector<complex> waves_to_samples(const std::vector<Wave> &waves, int n_samples) {
    std::vector<complex> samples(n_samples);
    for (size_t i = 0; i < samples.size(); ++i) {
        for (const auto &wave: waves) {
            samples[i] += wave.amplitude * std::sin(2 * M_PI * wave.frequency * (double) i / (double) n_samples +
                                                    wave.phase);
        }
    }
    return samples;
}

bool test(std::vector<complex> (*func)(const std::vector<complex> &)) {
    int n_waves = 10;

    std::vector<Wave> waves;
    waves.reserve(n_waves);

    for (int i = 0; i < n_waves; ++i) {
        waves.emplace_back(uniform_int(eng) % (NUM_SAMPLES / 2),
                           uniform_real(eng),
                           uniform_real(eng));
    }

    std::vector<complex> samples = waves_to_samples(waves, NUM_SAMPLES);
    std::vector<complex> X = func(samples);

    for (int i = 0; i < n_waves; ++i) {
        if (std::abs(waves[i].amplitude - std::abs(X[i])) > 0.0001) {
            return false;
        }
        if (std::abs(waves[i].frequency - i) > 0.0001) {
            return false;
        }
        if (std::abs(waves[i].phase - std::arg(X[i])) > 0.0001) {
            return false;
        }
    }

    return true;
}

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

    std::cout << "Testing DFT..." << std::endl;
    if (test(dft)) {
        std::cout << "DFT test passed!" << std::endl;
    } else {
        std::cout << "DFT test failed!" << std::endl;
    }

    return 0;
}
