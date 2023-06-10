#include "utils.hpp"
#include "test.hpp"

#define STB_IMAGE_IMPLEMENTATION

#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"


std::random_device rd;
std::uniform_int_distribution<> uniform(0, std::numeric_limits<int>::max());
std::mt19937_64 engine(rd());


int main() {

    std::vector<int64_t> coeff(10);
    std::vector<int64_t> coeff2(10);

    std::fill(coeff.begin(), coeff.end(), 1);
    std::fill(coeff2.begin(), coeff2.end(), 1);

    auto res = multiply(coeff, coeff2);

    for (auto &i : res) {
        std::cout << i << " ";
    }

    return 0;
}
