#include "utils.hpp"

std::vector<uint8_t> apply_sobel(uint8_t *img, int width, int height) {
    int sobel_x[3][3] = {{-1, 0, 1},
                         {-2, 0, 2},
                         {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1},
                         {0,  0,  0},
                         {1,  2,  1}};

    std::vector<uint8_t> edge_detected_image(width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gradient_x = 0;
            int gradient_y = 0;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    if (y + ky < 0 || y + ky >= height ||
                        x + kx < 0 || x + kx >= width)
                        continue;

                    uint8_t pixel_intensity = img[(y + ky) * width + (x + kx)];

                    gradient_x += pixel_intensity * sobel_x[ky + 1][kx + 1];
                    gradient_y += pixel_intensity * sobel_y[ky + 1][kx + 1];
                }
            }

            int gradient_magnitude = std::sqrt(gradient_x * gradient_x + gradient_y * gradient_y);

            edge_detected_image[y * width + x] = std::min(gradient_magnitude, 255);
        }
    }

    return edge_detected_image;
}
