#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <thread>

using complex = std::complex<double>;

std::vector<int> factorize(int n) {
    std::vector<int> factors;
    while (n % 2 == 0) {
        factors.push_back(2);
        n = n / 2;
    }

    for (int i = 3; i <= sqrt(n); i += 2) {
        while (n % i == 0) {
            factors.push_back(i);
            n = n / i;
        }
    }

    if (n > 2)
        factors.push_back(n);

    return factors;
}

std::vector<std::vector<complex>> decompose(const std::vector<complex> &x) {
    std::vector<std::vector<complex>> sub_sequences;

    int N = x.size();
    auto factors = factorize(N);

    for (auto factor: factors) {
        int num_sub_sequences = N / factor;
        for (int i = 0; i < num_sub_sequences; ++i) {
            std::vector<complex> sub_sequence(factor);
            for (int j = 0; j < factor; ++j) {
                sub_sequence[j] = x[i * factor + j];
            }
            sub_sequences.push_back(sub_sequence);
        }
    }

    return sub_sequences;
}

void fft(std::vector<complex> &x) {
    int N = x.size();
    if (N <= 1) return;

    std::vector<complex> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; ++i) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft(even);
    fft(odd);

    for (int k = 0; k < N / 2; ++k) {
        complex t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}

std::vector<std::complex<double>> mixed_radix_fft(const std::vector<double> &input) {
    int N = input.size();

    std::vector<std::complex<double>> complex_input(input.begin(), input.end());

    auto sub_sequences = decompose(complex_input);

    int num_cores = std::thread::hardware_concurrency();
    int num_threads = std::min(num_cores, static_cast<int>(sub_sequences.size()));
    std::vector<std::thread> threads(num_threads);

    int tasks_per_thread = sub_sequences.size() / num_threads;
    int extra_tasks = sub_sequences.size() % num_threads;

    auto work = [&sub_sequences](int start, int end) {
        // print start and end
        printf("start: %d, end: %d\n", start, end);
        for (int i = start; i < end; ++i) {
            fft(sub_sequences[i]);
        }
    };

    int start = 0;
    for (int i = 0; i < num_threads; ++i) {
        int end = start + tasks_per_thread;
        if (i < extra_tasks) ++end;
        threads[i] = std::thread(work, start, end);
        start = end;
    }

    for (auto &thread: threads) {
        thread.join();
    }

    // Twiddle factors multiplication and results combining
    std::vector<std::complex<double>> output(N);
    for (int k = 0; k < N; ++k) {
        for (size_t i = 0; i < sub_sequences.size(); ++i) {
            int m = N / sub_sequences[i].size();
            output[k] += std::polar(1.0, -2 * M_PI * k * m / N) * sub_sequences[i][k % m];
        }
    }

    return output;
}

int main() {
    std::vector<double> input;
    for (int i = 0; i < 500; ++i) {
        input.push_back(i);
    }
    auto output = mixed_radix_fft(input);

    // Print out the resulting FFT coefficients
//    for (const auto &coef: output) {
//        std::cout << coef << '\n';
//    }

    return 0;
}
