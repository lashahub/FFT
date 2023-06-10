#include "utils.hpp"


size_t next_power_of_2(size_t n) {
    size_t res = 1;
    while (res < n) {
        res <<= 1;
    }
    return res;
}

std::vector<complex> adjust_size(std::vector<complex> &P, int new_size) {
    std::vector<complex> P_new(new_size);
    for (int i = 0; i < P.size(); i++) {
        P_new[i] = P[i];
    }
    return P_new;
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

complex cmplx(double ang) {
    return complex(std::cos(ang), std::sin(ang));
}

void join_threads(std::vector<std::thread> &threads) {
    for (auto &thread : threads) {
        thread.join();
    }
    threads.clear();
}

void radix2_fft_sequential(std::vector<complex>& coeff, std::vector<complex>& eval) {
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

    // Recursive FFT calls
    radix2_fft_sequential(even, even);
    radix2_fft_sequential(odd, odd);

    double ang = 2 * M_PI / (double) n;

    // Sequentially combine
    for (size_t i = 0; 2 * i < n; i++) {
        complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
        coeff[i] = even[i] + t * odd[i];
        coeff[i + n / 2] = even[i] - t * odd[i];
    }
}


void fft_dit_inplace_seq(std::vector<complex>& P, size_t start, size_t stride, size_t N) {

    if (N <= 1) {
        return;
    }

    size_t p = get_prime_factor(N);
    size_t q = N / p;

    if (p == N) {

        std::vector<complex> temp(N);
        for (int i = 0; i < N; i++){
            temp[i] = P[start + i * stride];
        }
        temp = dft_seq(temp);
        for (int i = 0; i < N; i++){
            P[start + i * stride] = temp[i];
        }
        return;
    }

    for (int i = 0; i < p; i++) {
        fft_dit_inplace_seq(P, start + i * stride, stride * p, q);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P[start + stride * (j * p + i)] *= std::exp(complex(0, -2 * M_PI * i * j / N));
        }
    }

    for (int i = 0; i < q; i++) {
        fft_dit_inplace_seq(P, start + stride * i * p, stride, p);
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

std::vector<complex> ditfft_sequential(std::vector<complex> &P) {
    int N = P.size();
    int p = get_prime_factor(N);

    if (p == N) {
        return dft_seq(P);
    }

    int q = N / p;
    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            A[i][j] = P[j * p + i];
        }
    }

    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (int i = 0; i < p; ++i) {
        B[i] = ditfft_sequential(A[i]);
    }

    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            B[i][j] *= std::exp(complex(0, -2 * M_PI * i * j / N));
        }
    }

    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (int i = 0; i < q; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = B[j][i];
        }
    }

    for (int i = 0; i < q; i++) {
        C[i] = ditfft_sequential(C[i]);
    }

    std::vector<complex> P_star(N);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < q; j++) {
            P_star[i * q + j] = C[j][i];
        }
    }

    return P_star;
}

std::vector<complex> fft(std::vector<complex> &coeff) {
    size_t n = coeff.size();
    size_t np2 = next_power_of_2(n);
    coeff.resize(np2);
    std::vector<complex> eval(np2);
    radix2_fft_parallel(coeff, eval);
    eval.resize(n);
    return eval;
}

std::vector<complex> ifft(std::vector<complex> &coeff) {
    size_t n = coeff.size();
    for (auto &x: coeff) x = std::conj(x);

    std::vector<complex> eval(n);
    eval = fft(coeff);

    for (auto &x: eval) {
        x = std::conj(x);
        x /= (double) n;
    }

    return eval;
}

std::vector<complex> dft_seq(std::vector<complex> &a, bool power_of_2) {
    size_t N = a.size();
    if (power_of_2){
        N = next_power_of_2(a.size());
    }
    std::vector<complex> eval(a.size());
    for (size_t k = 0; k < a.size(); k++) {
        for (size_t j = 0; j < a.size(); j++) {
            eval[k] += a[j] * std::exp(complex(0, -2 * M_PI * (double) j * (double) k / (double) N));
        }
    }
    return eval;
}

void radix2_fft_parallel(std::vector<complex> &coeff, std::vector<complex> &eval, size_t depth) {
    size_t n = coeff.size();
    if (n == 1)
        return;

    std::vector<complex> even(n / 2), odd(n / 2);

    if (depth < MAX_THREAD_DEPTH) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 2; i++) {
            even[i] = coeff[2 * i];
            if (2 * i + 1 < n) {
                odd[i] = coeff[2 * i + 1];
            }
        }
    } else {
        for (size_t i = 0; 2 * i < n; i++) {
            even[i] = coeff[2 * i];
            if (2 * i + 1 < n) {
                odd[i] = coeff[2 * i + 1];
            }
        }
    }

    if (depth < MAX_THREAD_DEPTH) {
        std::thread t1(radix2_fft_parallel, std::ref(even), std::ref(even), depth + 1);
        std::thread t2(radix2_fft_parallel, std::ref(odd), std::ref(odd), depth + 1);
        t1.join();
        t2.join();
    } else {
        radix2_fft_parallel(even, even, depth + 1);
        radix2_fft_parallel(odd, even, depth + 1);
    }

    double ang = 2 * M_PI / (double) n;

    if (depth < MAX_THREAD_DEPTH) {
#pragma omp parallel for
        for (size_t i = 0; i < n / 2; i++) {
            complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
            coeff[i] = even[i] + t * odd[i];
            coeff[i + n / 2] = even[i] - t * odd[i];
        }
    } else {
        for (size_t i = 0; 2 * i < n; i++) {
            complex t(std::cos((double) i * ang), (std::sin((double) i * ang)));
            eval[i] = even[i] + t * odd[i];
            eval[i + n / 2] = even[i] - t * odd[i];
        }
    }
}

std::vector<complex> ditfft_parallel(std::vector<complex> &P_OLD, size_t num_threads) {

    std::vector<complex> P(P_OLD.size());
    for (size_t i = 0; i < P_OLD.size(); i++) {
        P[i] = P_OLD[i];
    }
    size_t N = modify_size(P_OLD.size(), num_threads);
    P.resize(N);

    size_t p = num_threads;
    size_t q = N / p;

    std::vector<std::thread> threads;

    std::vector<std::vector<complex>> A(p, std::vector<complex>(q));
    for (size_t i = 0; i < p - 1; i++) {
        threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < q; j++) {
                A[i][j] = P[j * p + i];
            }
        }));
    }
    for (size_t j = 0; j < q; j++) {
        A[p - 1][j] = P[j * p + p - 1];
    }

    join_threads(threads);

    std::vector<std::vector<complex>> B(p, std::vector<complex>(q));
    for (size_t i = 0; i < p - 1; i++) {
        threads.push_back(std::thread([&, i]() {
            fft_dit_inplace_seq(A[i], 0, 1, q);
        }));
    }
    fft_dit_inplace_seq(A[p - 1], 0, 1, q);

    join_threads(threads);

    for (size_t i = 0; i < p-1; i++) {
        threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < q; j++) {
                A[i][j] *= std::exp(complex(0, -2 * M_PI * i * j / N));
            }
        }));
    }
    for (size_t j = 0; j < q; j++) {
        A[p - 1][j] *= std::exp(complex(0, -2 * M_PI * (p - 1) * j / N));
    }

    join_threads(threads);

    std::vector<std::vector<complex>> C(q, std::vector<complex>(p));
    for (size_t j = 0; j < p - 1; j++){
        threads.push_back(std::thread([&, j]() {
            for (size_t i = 0; i < q; i++) {
                C[i][j] = A[j][i];
            }
        }));
    }
    for (size_t i = 0; i < q; i++) {
        C[i][p - 1] = A[p - 1][i];
    }
    join_threads(threads);

    size_t chunk_size = (q + p - 1) / p;
    for (size_t t = 0; t < p - 1; ++t) {
        threads.push_back(std::thread([&, t] {
            size_t start = t * chunk_size;
            size_t end = (t+1) * chunk_size;
            for (size_t i = start; i < end; ++i) {
                fft_dit_inplace_seq(C[i], 0, 1, p);
            }
        }));
    }
    size_t start = (p - 1) * chunk_size;
    size_t end = q;
    for (size_t i = start; i < end; ++i) {
        fft_dit_inplace_seq(C[i], 0, 1, p);
    }

    join_threads(threads);


    std::vector<complex> P_star(N);
    for (size_t i = 0; i < p - 1; i++) {
        threads.push_back(std::thread([&, i]() {
            for (int j = 0; j < q; j++) {
                P_star[i * q + j] = C[j][i];
            }
        }));
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
