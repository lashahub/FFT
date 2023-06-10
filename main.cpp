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

    std::cout << "Testing correctness..." << std::endl;
    if (test_all_correctness()) {
        std::cout << "Correctness test passed!" << std::endl;
    } else {
        std::cout << "Correctness test failed!" << std::endl;
    }

    return 0;
}
