#include "bmp.hpp"

BMP::~BMP() {
    delete[] data;
}

bool read_bmp(const std::string &file_name, BMP &bmp) {
    std::ifstream inp{file_name, std::ios_base::binary};
    if (inp) {
        inp.read((char *) &bmp.file_header, sizeof(bmp.file_header));
        if (bmp.file_header.file_type != 0x4D42) {
            return false;
        }

        inp.read((char *) &bmp.bmp_info_header, sizeof(bmp.bmp_info_header));
        bmp.data = new uint8_t[bmp.bmp_info_header.size_image];
        inp.seekg(bmp.file_header.offset_data, std::ifstream::beg);

        inp.read((char *) bmp.data, bmp.bmp_info_header.size_image);
        uint8_t temp;

        for (uint32_t i = 0; i < bmp.bmp_info_header.size_image; i += 3) {
            temp = bmp.data[i];
            bmp.data[i] = bmp.data[i + 2];
            bmp.data[i + 2] = temp;
        }
    } else {
        return false;
    }

    return true;
}
