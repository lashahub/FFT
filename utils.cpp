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


std::vector<Point> transformContour(const std::vector<Point> &contour) {
    std::vector<complex> contour_x(contour.size());
    std::vector<complex> contour_y(contour.size());

    for (size_t i = 0; i < contour.size(); i++) {
        contour_x[i] = complex(contour[i].x, 0);
        contour_y[i] = complex(contour[i].y, 0);
    }

    fft(contour_x);
    fft(contour_y);

    ifft(contour_x);
    ifft(contour_y);

    std::vector<Point> contour_transformed(contour.size());
    for (size_t i = 0; i < contour.size(); i++) {
        contour_transformed[i].x = (int) contour_x[i].real();
        contour_transformed[i].y = (int) contour_y[i].real();
    }

    return contour_transformed;
}
