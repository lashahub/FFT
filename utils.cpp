#include "utils.hpp"


size_t next_power_of_2(size_t n) {
    size_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}


void fft_rec(std::vector<complex> &a, size_t depth) {
    size_t n = a.size();
    if (n == 1)
        return;

    std::vector<complex> even(n / 2), odd(n / 2);

    if (depth < MAX_THREAD_DEPTH) {
#pragma omp parallel for
        for (int i = 0; i < n / 2; i++) {
            even[i] = a[2 * i];
            if (2 * i + 1 < n) {
                odd[i] = a[2 * i + 1];
            }
        }
    } else {
        for (size_t i = 0; 2 * i < n; i++) {
            even[i] = a[2 * i];
            if (2 * i + 1 < n) {
                odd[i] = a[2 * i + 1];
            }
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

    if (depth < MAX_THREAD_DEPTH) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 2; i++) {
            complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
            a[i] = even[i] + t * odd[i];
            a[i + n / 2] = even[i] - t * odd[i];
        }
    } else {
        for (size_t i = 0; 2 * i < n; i++) {
            complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
            a[i] = even[i] + t * odd[i];
            a[i + n / 2] = even[i] - t * odd[i];
        }
    }
}


void fft(std::vector<complex> &a) {
    size_t n = a.size();
    a.resize(next_power_of_2(n));
    fft_rec(a);
    a.resize(n);
}

void ifft(std::vector<complex> &a) {
    size_t n = a.size();
    for (auto &x: a) x = std::conj(x);

    fft(a);

    for (auto &x: a) {
        x = std::conj(x);
        x /= (double) n;
    }
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
    // Convert the contour to a list of complex numbers
    std::vector<complex> contour_x(contour.size());
    std::vector<complex> contour_y(contour.size());

    for (int i = 0; i < contour.size(); i++) {
        contour_x[i] = complex(contour[i].x, 0);
        contour_y[i] = complex(contour[i].y, 0);
    }

    // Perform FFT on contour
    fft(contour_x);
    fft(contour_y);

    printDesmosEquations(contour_x, contour_y, 5);

    // Truncate the transformed contour to the desired number of frequencies
    if (contour_x.size() > numFrequencies) {
        std::fill(contour_x.begin() + numFrequencies, contour_x.end(), 0);
        std::fill(contour_y.begin() + numFrequencies, contour_y.end(), 0);
    }

    // Perform iFFT on the transformed contour
    ifft(contour_x);
    ifft(contour_y);

    // Convert back to a list of points
    std::vector<Point> contour_transformed(contour.size());
    for (int i = 0; i < contour.size(); i++) {
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
    int n = a.size();
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        int64_t wlen = invert ? power(root, mod - 1 - (mod - 1) / len, mod) : power(root, (mod - 1) / len, mod);
        for (int i = 0; i < n; i += len) {
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
    std::vector<int64_t> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < a.size() + b.size()) n <<= 1;
    fa.resize(n);
    fb.resize(n);
    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++) fa[i] = (fa[i] * fb[i]) % mod;
    fft(fa, true);
    return fa;
}

/*
 * Audio processing
 */

WavData readWavFile(const std::string &filename) {
    WavHeader header;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cout << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    file.read((char *) &header, sizeof(WavHeader));

    // Check the audio format to ensure it is PCM
    if (header.audioFormat != 1) {
        std::cout << "Unsupported audio format: " << header.audioFormat << std::endl;
        exit(1);
    }

    // Check the sample size to ensure it is 16 bit
    if (header.bitsPerSample != 16) {
        std::cout << "Unsupported bit depth: " << header.bitsPerSample << std::endl;
        exit(1);
    }

    // Allocate a vector for the audio data
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
