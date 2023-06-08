#include "utils.hpp"

enum class MODE {
    DFT_SEQ,
    DFT_PAR,
    FFT_R2_SEQ,
    FFT_R2_PAR,
    FFT_DIT_SEQ,
    FFT_DIT_PAR
};

// Auxiliary functions --------------------------------------------------------

size_t next_power_of_2(size_t n) {
    size_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}

size_t get_prime_factor(size_t N) {
    std::vector<size_t> factors;
    size_t p = 2;
    while (N > 1) {
        if (N % p == 0) {
            break;
        }
        p++;
    }
    return p;
}

size_t modify_size(size_t N, size_t p) {
    size_t N_star = p;
    while (N_star < N) {
        N_star <<= 1;
    }
    return N_star;
}

void join_threads(std::vector<std::thread> &threads) {
    for (auto &thread: threads) {
        thread.join();
    }
    threads.clear();
}

// FFT functions --------------------------------------------------------------

std::vector<complex> dft_seq(std::vector<complex> &coeff, bool power_of_2) {
    size_t N = coeff.size();
    if (power_of_2) {
        N = next_power_of_2(N);
    }
    std::vector<complex> eval(coeff.size());
    for (size_t k = 0; k < coeff.size(); k++) {
        for (size_t j = 0; j < coeff.size(); j++) {
            eval[k] += coeff[j] * std::exp(complex(0, 2 * M_PI * (double) j * (double) k / (double) N));
        }
    }
    return eval;
}

std::vector<complex> dft_par(std::vector<complex> &coeff, bool power_of_2, size_t THREADS_AVAILABLE) {
    size_t N = coeff.size();
    if (power_of_2) {
        N = next_power_of_2(N);
    }

    std::vector<std::thread> threads;
    threads.reserve(THREADS_AVAILABLE - 1);

    std::vector<complex> eval(coeff.size());

    for (size_t i = 1; i < THREADS_AVAILABLE; i++) {
        threads.emplace_back([i, &coeff, &eval, N, THREADS_AVAILABLE]() {
            for (size_t k = i; k < coeff.size(); k += THREADS_AVAILABLE) {
                for (size_t j = 0; j < coeff.size(); j++) {
                    eval[k] += coeff[j] * std::exp(complex(0, 2 * M_PI * (double) j * (double) k / (double) N));
                }
            }
        });
    }
    for (size_t k = 0; k < coeff.size(); k += THREADS_AVAILABLE) {
        for (size_t j = 0; j < coeff.size(); j++) {
            eval[k] += coeff[j] * std::exp(complex(0, 2 * M_PI * (double) j * (double) k / (double) N));
        }
    }

    join_threads(threads);

    return eval;
}

void radix2_fft_sequential(std::vector<complex> &coeff) {
    size_t n = coeff.size();
    if (n == 1)
        return;

    std::vector<complex> even(n / 2), odd(n / 2);

    for (size_t i = 0; 2 * i < n; i++) {
        even[i] = coeff[2 * i];
        if (2 * i + 1 < n) {
            odd[i] = coeff[2 * i + 1];
        }
    }

    radix2_fft_sequential(even);
    radix2_fft_sequential(odd);

    double ang = 2 * M_PI / (double) n;

    for (size_t i = 0; 2 * i < n; i++) {
        complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
        coeff[i] = even[i] + t * odd[i];
        coeff[i + n / 2] = even[i] - t * odd[i];
    }
}

void radix2_fft_parallel(std::vector<complex> &coeff, size_t THREADS_AVAILABLE) {
    size_t n = coeff.size();
    if (n == 1)
        return;

    std::vector<std::thread> threads;
    threads.reserve(THREADS_AVAILABLE - 1);

    std::vector<complex> even(n / 2), odd(n / 2);

    for (size_t t_i = 1; t_i < THREADS_AVAILABLE; t_i++) {
        threads.emplace_back([&, t_i] {
            for (size_t i = t_i; 2 * i < n; i += THREADS_AVAILABLE) {
                even[i] = coeff[2 * i];
                if (2 * i + 1 < n) {
                    odd[i] = coeff[2 * i + 1];
                }
            }
        });
    }
    for (size_t i = 0; 2 * i < n; i += THREADS_AVAILABLE) {
        even[i] = coeff[2 * i];
        if (2 * i + 1 < n) {
            odd[i] = coeff[2 * i + 1];
        }
    }
    join_threads(threads);

    if (THREADS_AVAILABLE > 1) {
        std::thread t1(radix2_fft_parallel, std::ref(even), THREADS_AVAILABLE / 2);
        std::thread t2(radix2_fft_parallel, std::ref(odd), THREADS_AVAILABLE / 2);
        t1.join();
        t2.join();
    } else {
        radix2_fft_parallel(even);
        radix2_fft_parallel(odd);
    }

    double ang = 2 * M_PI / (double) n;

    for (size_t t_i = 1; t_i < THREADS_AVAILABLE; t_i++) {
        threads.emplace_back([&, t_i] {
            for (size_t i = t_i; 2 * i < n; i += THREADS_AVAILABLE) {
                complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
                coeff[i] = even[i] + t * odd[i];
                coeff[i + n / 2] = even[i] - t * odd[i];
            }
        });
    }
    for (size_t i = 0; 2 * i < n; i += THREADS_AVAILABLE) {
        complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
        coeff[i] = even[i] + t * odd[i];
        coeff[i + n / 2] = even[i] - t * odd[i];
    }
    join_threads(threads);
}


void dit_fft_inplace_par(std::vector<complex> &P, size_t start, size_t stride, size_t N, size_t THREADS_AVAILABLE = 1) {
    if (N <= 1) {
        return;
    }

    std::vector<std::thread> threads;

    size_t p = get_prime_factor(N);
    size_t q = N / p;

    if (p == N) {
        std::vector<complex> temp(N);
        for (int i = 0; i < N; i++) {
            temp[i] = P[start + i * stride];
        }
        temp = dft_seq(temp);
        for (int i = 0; i < N; i++) {
            P[start + i * stride] = temp[i];
        }
        return;
    }

    if (THREADS_AVAILABLE == 1) {
        dit_fft_inplace_seq(P, start, stride, q);
    } else if (p <= THREADS_AVAILABLE) {
        for (size_t i = 0; i < p; i++) {
            threads.emplace_back([&, i] {
                dit_fft_inplace_par(P, start + i * stride, stride * p, q, 1);
            });
        }
        join_threads(threads);
    } else {
        size_t tasks_per_thread = (p + THREADS_AVAILABLE - 1) / THREADS_AVAILABLE;
        for (int i = 0; i < THREADS_AVAILABLE; i++) {
            threads.emplace_back([&, i] {
                for (size_t j = i * tasks_per_thread; j < std::min((i + 1) * tasks_per_thread, p); j++) {
                    dit_fft_inplace_par(P, start + j * stride, stride * p, q, (THREADS_AVAILABLE + p - 1) / p);
                }
            });
        }
        join_threads(threads);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P[start + stride * (j * p + i)] *= std::exp(complex(0, 2 * M_PI * i * j / (double) N));
        }
    }

    if (THREADS_AVAILABLE == 1) {
        dit_fft_inplace_seq(P, start, stride, p);
    } else if (q <= THREADS_AVAILABLE) {
        for (size_t i = 0; i < q; i++) {
            threads.emplace_back([&, i] {
                dit_fft_inplace_par(P, start + i * stride, stride * q, p, 1);
            });
        }
        join_threads(threads);
    } else {
        size_t tasks_per_thread = (q + THREADS_AVAILABLE - 1) / THREADS_AVAILABLE;
        for (int i = 0; i < THREADS_AVAILABLE; i++) {
            threads.emplace_back([&, i] {
                for (size_t j = i * tasks_per_thread; j < std::min((i + 1) * tasks_per_thread, q); j++) {
                    dit_fft_inplace_par(P, start + j * stride, stride * q, p, (THREADS_AVAILABLE + q - 1) / q);
                }
            });
        }
        join_threads(threads);
    }

    std::vector<complex> temp(p * q);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            temp[i * q + j] = P[start + stride * (j * p + i)];
        }
    }

    for (int i = 0; i < p * q; i++) {
        P[start + stride * i] = temp[i];
    }
}


void dit_fft_inplace_seq(std::vector<complex> &P, size_t start, size_t stride, size_t N) {
    if (N <= 1) {
        return;
    }

    size_t p = get_prime_factor(N);
    size_t q = N / p;

    if (p == N) {
        std::vector<complex> temp(N);
        for (int i = 0; i < N; i++) {
            temp[i] = P[start + i * stride];
        }
        temp = dft_seq(temp);
        for (int i = 0; i < N; i++) {
            P[start + i * stride] = temp[i];
        }
        return;
    }

    for (int i = 0; i < p; i++) {
        dit_fft_inplace_seq(P, start + i * stride, stride * p, q);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P[start + stride * (j * p + i)] *= std::exp(complex(0, 2 * M_PI * i * j / (double) N));
        }
    }

    for (int i = 0; i < q; i++) {
        dit_fft_inplace_seq(P, start + stride * i * p, stride, p);
    }

    std::vector<complex> temp(p * q);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            temp[i * q + j] = P[start + stride * (j * p + i)];
        }
    }

    for (int i = 0; i < p * q; i++) {
        P[start + stride * i] = temp[i];
    }
}

std::vector<complex> dit_fft_sequential(std::vector<complex> &P) {
    size_t N = P.size();
    size_t p = get_prime_factor(N);

    if (p == N) {
        return dft_seq(P);
    }

    size_t q = N / p;
    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            A[i][j] = P[j * p + i];
        }
    }

    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (int i = 0; i < p; ++i) {
        B[i] = dit_fft_sequential(A[i]);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            B[i][j] *= std::exp(complex(0, 2 * M_PI * i * j / (double) N));
        }
    }

    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = B[j][i];
        }
    }

    for (int i = 0; i < q; i++) {
        C[i] = dit_fft_sequential(C[i]);
    }

    std::vector<complex> P_star(N);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P_star[i * q + j] = C[j][i];
        }
    }

    return P_star;
}

//

std::vector<complex> fft(std::vector<complex> &coeff) {
    size_t n = coeff.size();
    size_t np2 = next_power_of_2(n);
    coeff.resize(np2);
    std::vector<complex> eval(np2);
    radix2_fft_parallel(coeff);
    eval.resize(n);
    return eval;
}

std::vector<complex> ifft(std::vector<complex> &coeff) {
    size_t n = coeff.size();
    for (auto &x: coeff) x = std::conj(x);

    std::vector<complex> eval(n);
    eval = fft(coeff);

    for (auto &x: eval) {
        x = std::conj(x) / (double) n;
    }

    return eval;
}


std::vector<complex> dit_fft_parallel(std::vector<complex> &P_OLD, size_t num_threads) {

    std::vector<complex> P(P_OLD.size());
    for (size_t i = 0; i < P_OLD.size(); i++) {
        P[i] = P_OLD[i];
    }
    size_t N = modify_size(P_OLD.size(), num_threads);
    P.resize(N);

    size_t p = num_threads;
    size_t q = N / p;

    std::vector<std::thread> threads;
    threads.reserve(p);

    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (size_t i = 0; i < p - 1; i++) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < q; j++) {
                A[i][j] = P[j * p + i];
            }
        });
    }
    for (size_t j = 0; j < q; j++) {
        A[p - 1][j] = P[j * p + p - 1];
    }
    join_threads(threads);

    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (size_t i = 0; i < p - 1; i++) {
        threads.emplace_back([&, i]() {
            dit_fft_inplace_seq(A[i], 0, 1, q);
        });
    }
    dit_fft_inplace_seq(A[p - 1], 0, 1, q);
    join_threads(threads);

    for (size_t i = 0; i < p - 1; i++) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < q; j++) {
                A[i][j] *= std::exp(complex(0, 2 * M_PI * (double) i * j / (double) N));
            }
        });
    }
    for (size_t j = 0; j < q; j++) {
        A[p - 1][j] *= std::exp(complex(0, 2 * M_PI * (double) (p - 1) * (double) j / (double) N));
    }
    join_threads(threads);

    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (size_t j = 0; j < p - 1; j++) {
        threads.emplace_back([&, j]() {
            for (size_t i = 0; i < q; i++) {
                C[i][j] = A[j][i];
            }
        });
    }
    for (size_t i = 0; i < q; i++) {
        C[i][p - 1] = A[p - 1][i];
    }
    join_threads(threads);

    size_t chunk_size = (q + p - 1) / p;
    for (size_t t = 0; t < p - 1; ++t) {
        threads.emplace_back([&, t] {
            size_t start = t * chunk_size;
            size_t end = (t + 1) * chunk_size;
            for (size_t i = start; i < end; ++i) {
                dit_fft_inplace_seq(C[i], 0, 1, p);
            }
        });
    }
    size_t start = (p - 1) * chunk_size;
    size_t end = q;
    for (size_t i = start; i < end; ++i) {
        dit_fft_inplace_seq(C[i], 0, 1, p);
    }

    join_threads(threads);


    std::vector<complex> P_star(N);
    for (size_t i = 0; i < p - 1; i++) {
        threads.emplace_back([&, i]() {
            for (int j = 0; j < q; j++) {
                P_star[i * q + j] = C[j][i];
            }
        });
    }
    for (size_t j = 0; j < q; j++) {
        P_star[(p - 1) * q + j] = C[j][p - 1];
    }

    join_threads(threads);

    return P_star;
}


//********************************* Curves *********************************//

Point findStartingPoint(const uint8_t *image, int width, int height, int channels) {
    const int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
    const int dy[8] = {-1, -1, 0, 1, 1, 1, 0, -1};

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (image[(y * width + x) * channels] == 0) {
                int count = 0;
                for (int i = 0; i < 8; i++) {
                    int nx = x + dx[i];
                    int ny = y + dy[i];
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && image[(ny * width + nx) * channels] == 0) {
                        count++;
                    }
                }
                if (count == 1) {
                    return {x, y};
                }
            }
        }
    }

    return {-1, -1};
}


std::vector<Point> traceContour(const uint8_t *image, int width, int height, int channels, Point start) {
    const int dx[8] = {0, 1, 0, -1, -1, -1, 1, 1};
    const int dy[8] = {-1, 0, 1, 0, -1, 1, -1, 1};
    std::vector<Point> contour;

    if (start.x == -1 || start.y == -1) {
        std::cout << "Invalid start point!" << std::endl;
        return contour;
    }

    std::vector<bool> visited(width * height, false);

    int x = start.x;
    int y = start.y;
    while (true) {
        contour.push_back({x, y});
        visited[y * width + x] = true;
        bool found = false;
        for (int i = 0; i < 8; i++) {
            int nx = x + dx[i];
            int ny = y + dy[i];
            if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                image[(ny * width + nx) * channels] == 0 && !visited[ny * width + nx]) {
                x = nx;
                y = ny;
                found = true;
                break;
            }
        }
        if (!found || (x == start.x && y == start.y)) {
            break;
        }
    }

    return contour;
}


std::vector<Point> transformContour(const std::vector<Point> &contour) {
    std::vector<complex> contour_x(contour.size());
    std::vector<complex> contour_y(contour.size());

    for (size_t i = 0; i < contour.size(); i++) {
        contour_x[i] = complex(contour[i].x, 0);
        contour_y[i] = complex(contour[i].y, 0);
    }

    contour_x = fft(contour_x);
    contour_y = fft(contour_y);

    contour_x = ifft(contour_x);
    contour_y = ifft(contour_y);

    std::vector<Point> contour_transformed(contour.size());
    for (size_t i = 0; i < contour.size(); i++) {
        contour_transformed[i].x = (int) contour_x[i].real();
        contour_transformed[i].y = (int) contour_y[i].real();
    }

    return contour_transformed;
}

void
printDesmosEquations(const std::vector<complex> &contour_x, const std::vector<complex> &contour_y, int numFrequencies) {
    std::ofstream desmosFile;
    desmosFile.open("desmos.txt");
    desmosFile << "x(t) = ";
    for (int n = 0; n < numFrequencies; n++) {
        if (n != 0)
            desmosFile << " + ";
        desmosFile << std::fixed << std::setprecision(3) << contour_x[n].real()
                   << "*cos(2π*" << n << "*t/" << contour_x.size() << ")"
                   << " - " << contour_x[n].imag() << "*sin(2π*" << n << "*t/" << contour_x.size() << ")";
    }

    desmosFile << "\ny(t) = ";
    for (int n = 0; n < numFrequencies; n++) {
        if (n != 0)
            desmosFile << " + ";
        desmosFile << std::fixed << std::setprecision(3) << contour_y[n].real()
                   << "*sin(2π*" << n << "*t/" << contour_y.size() << ")"
                   << " + " << contour_y[n].imag() << "*cos(2π*" << n << "*t/" << contour_y.size() << ")";
    }
    desmosFile.close();
}


/*
 * Polynomial multiplication
 */


int64_t power(int64_t a, int64_t b, int64_t p) {
    int64_t x = 1, y = a;
    while (b > 0) {
        if (b % 2 == 1) {
            x = (x * y) % p;
        }
        y = (y * y) % p;
        b /= 2;
    }
    return x % p;
}


void fft(std::vector<int64_t> &a, bool invert) {
    size_t n = a.size();
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (size_t len = 2; len <= n; len <<= 1) {
        int64_t wlen = invert ? power(root, mod - 1 - (mod - 1) / len, mod) : power(root, (mod - 1) / len, mod);
        for (size_t i = 0; i < n; i += len) {
            int64_t w = 1;
            for (int j = 0; j < len / 2; ++j) {
                int64_t u = a[i + j], v = (a[i + j + len / 2] * w) % mod;
                a[i + j] = u + v < mod ? u + v : u + v - mod;
                a[i + j + len / 2] = u - v >= 0 ? u - v : u - v + mod;
                w = (w * wlen) % mod;
            }
        }
    }
    if (invert) {
        int64_t nrev = power(n, mod - 2, mod);
        for (int i = 0; i < n; ++i) a[i] = (a[i] * nrev) % mod;
    }
}


std::vector<int64_t> multiply(std::vector<int64_t> const &a, std::vector<int64_t> const &b) {
    std::vector<int64_t> P(a.begin(), a.end()), Q(b.begin(), b.end());
    size_t totalSize = a.size() + b.size();
    size_t n = next_power_of_2(totalSize);
    P.resize(n);
    Q.resize(n);
    fft(P, false);
    fft(Q, false);
    for (int i = 0; i < n; i++) P[i] = (P[i] * Q[i]) % mod;
    fft(P, true);
    return P;
}

// Audio processing -----------------------------------------------------------

WavData readWavFile(const std::string &filename) {
    WavHeader header{};
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    file.read((char *) &header, sizeof(WavHeader));

    if (header.audioFormat != 1) {
        std::cout << "Unsupported audio format: " << header.audioFormat << std::endl;
        exit(1);
    }

    if (header.bitsPerSample != 16) {
        std::cout << "Unsupported bit depth: " << header.bitsPerSample << std::endl;
        exit(1);
    }

    std::vector<uint16_t> audioData(header.subchunk2Size / 2);
    file.read(reinterpret_cast<char *>(audioData.data()), header.subchunk2Size);

    WavData wavData;
    wavData.audioFormat = header.audioFormat;
    wavData.numChannels = header.numChannels;
    wavData.sampleRate = header.sampleRate;
    wavData.bitsPerSample = header.bitsPerSample;

    if (header.numChannels == 1) {
        wavData.leftChannel = std::move(audioData);
    } else if (header.numChannels == 2) {
        for (size_t i = 0; i < audioData.size(); i += 2) {
            wavData.leftChannel.push_back(audioData[i]);
            wavData.rightChannel.push_back(audioData[i + 1]);
        }
    } else {
        std::cout << "Unsupported number of channels: " << header.numChannels << std::endl;
        exit(1);
    }

    return wavData;
}
