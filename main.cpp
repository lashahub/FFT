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

    int width, height, channels;

    uint8_t *img = stbi_load("we_love_cse305.png", &width, &height, &channels, 0);

    if (img == nullptr) {
        printf("Error in loading the image\n");
        return -1;
    }

    Point start = findStartingPoint(img, width, height, channels);
    std::vector<Point> contour = traceContour(img, width, height, channels, start);
    std::vector<Point> transformedContour = transformContour(contour, 5000);

    auto *output_img = new uint8_t[width * height * channels];
    memset(output_img, 255, width * height * channels);
    for (const Point &point: transformedContour) {
        if (point.x >= 0 && point.x < width && point.y >= 0 && point.y < height) {
            output_img[(point.y * width + point.x) * channels + 0] = 0;
            output_img[(point.y * width + point.x) * channels + 1] = 0;
            output_img[(point.y * width + point.x) * channels + 2] = 0;
        }
    }
    stbi_write_png("output.png", width, height, channels, output_img, width * channels);

    stbi_image_free(img);
    delete[] output_img;

    return 0;
}
