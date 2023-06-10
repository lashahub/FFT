#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#include "utils.hpp"
#include "test.hpp"

std::random_device rd;
std::uniform_int_distribution<> uniform(0, std::numeric_limits<int>::max());
std::mt19937_64 engine(rd());

using complex = std::complex<double>;


int main() {

//    if (test_all_correctness()) {
//        std::cout << "Correctness test passed" << std::endl;
//    } else {
//        std::cout << "Correctness test failed" << std::endl;
//    }

    std::vector<MODE> modes = {MODE::FFT_RADIX2_PAR};
    std::vector<size_t> num_threads = {16};
    std::vector<size_t> N = {1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000, 10000000, 11000000,
                             12000000, 13000000, 14000000, 15000000, 16000000, 17000000, 18000000, 19000000, 20000000};

    auto results = benchmark(modes, num_threads, N);

    std::cout << "Benchmark results:" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "Mode: " << i << std::endl;
        for (size_t j = 0; j < num_threads.size(); ++j) {
            std::cout << "Number of threads: " << num_threads[j] << std::endl;
            for (size_t k = 0; k < N.size(); ++k) {
                std::cout << N[k] << "," << results[i][j][k] << std::endl;
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
