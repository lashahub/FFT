#include "utils.hpp"


// Auxiliary functions --------------------------------------------------------

size_t next_power_of_2(size_t n) {
    size_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}


size_t get_prime_factor(size_t N) {
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


complex cmplx(double angle) {
    return std::exp(complex(0, angle));
//    return complex(std::cos(angle), std::sin(angle));
}


void join_threads(std::vector<std::thread> &threads) {
    for (auto &thread: threads) {
        thread.join();
    }
    threads.clear();
}


size_t new_N(MODE mode, size_t num_threads, size_t N) {
    if (mode == MODE::FFT_RADIX2_SEQ || mode == MODE::FFT_RADIX2_PAR) {
        return next_power_of_2(N);
    } else if (mode == MODE::FFT_DIT_PAR) {
        return modify_size(N, num_threads);
    }
    return N;
}


size_t r_int() {
    return uniform(engine);
}


// Main FFT -------------------------------------------------------------------


std::vector<complex> fft(MODE mode, std::vector<complex> &coeff, size_t num_threads) {
    if (coeff.empty()) {
        return coeff;
    }
    auto N = new_N(mode, num_threads, coeff.size());
    coeff.resize(N);
    switch (mode) {
        case MODE::DFT_SEQ:
            return dft_seq(coeff, N);
        case MODE::DFT_PAR:
            return dft_par(coeff, num_threads, N);
        case MODE::FFT_RADIX2_SEQ:
            fft_radix2_seq(coeff);
            return coeff;
        case MODE::FFT_RADIX2_PAR:
            fft_radix2_par(coeff, num_threads);
            return coeff;
        case MODE::FFT_DIT_INPLACE_SEQ:
            fft_dit_inplace_seq(coeff, 0, 1, coeff.size());
            return coeff;
        case MODE::FFT_DIT_INPLACE_PAR:
            fft_dit_inplace_par(coeff, 0, 1, coeff.size(), num_threads);
            return coeff;
        case MODE::FFT_DIT_SEQ:
            return fft_dit_seq(coeff);
        case MODE::FFT_DIT_PAR:
            fft_dit_par(coeff, 0, 1, coeff.size(), num_threads);
            return coeff;
    }
    return coeff;
}

void ifft(MODE mode, std::vector<complex> &coeff, size_t num_threads) {
    size_t n = coeff.size();
    for (auto &x: coeff) x = std::conj(x);

    fft(mode, coeff, num_threads);

    for (auto &x: coeff) {
        x = std::conj(x);
        x /= (double) n;
    }
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


std::vector<Point> transformContour(const std::vector<Point> &contour, int numFrequencies) {
    std::vector<complex> contour_x(contour.size());
    std::vector<complex> contour_y(contour.size());

    for (size_t i = 0; i < contour.size(); i++) {
        contour_x[i] = complex(contour[i].x, 0);
        contour_y[i] = complex(contour[i].y, 0);
    }

    contour_x = fft(MODE::FFT_RADIX2_PAR, contour_x, 8);
    contour_y = fft(MODE::FFT_RADIX2_PAR, contour_y, 8);

    int cutOffIndex = numFrequencies;
    for (size_t i = cutOffIndex; i < contour_x.size() / 2; i++) {
        contour_x[i] = 0;
        contour_x[contour_x.size() - i] = 0; // for symmetry
    }

    for (size_t i = cutOffIndex; i < contour_y.size() / 2; i++) {
        contour_y[i] = 0;
        contour_y[contour_y.size() - i] = 0; // for symmetry
    }

    ifft(MODE::FFT_RADIX2_PAR, contour_x, 8);
    ifft(MODE::FFT_RADIX2_PAR, contour_y, 8);

    std::vector<Point> contour_transformed(contour.size());
    for (size_t i = 0; i < contour.size(); i++) {
        contour_transformed[i].x = (int) contour_x[i].real();
        contour_transformed[i].y = (int) contour_y[i].real();
    }

    return contour_transformed;
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
