#include <iostream>
#include <complex>
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

    std::vector<Wave> waves = {
            {440,  1,    0},
            {880,  0.5,  0},
            {1320, 0.25, 0}
    };



    std::vector<complex> X = dft(WavesToSamples(waves, 44100, 0.1));
    for (size_t i = 0; i < 500; ++i) {
        std::cout << i << " " << std::norm(X[i]) << std::endl;
    }
    return 0;
}
