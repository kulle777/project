size_t idx(size_t x, size_t y, size_t width, size_t height, int xoff, int yoff) {
    size_t resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    size_t resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

__kernel
void sobel3x3(const uint8_t *restrict in, size_t width, size_t height,
    int16_t *restrict output_x, int16_t *restrict output_y) {
    // LOOP 1.1
    for (size_t y = 0; y < height; y++) {
        // LOOP 1.2
        for (size_t x = 0; x < width; x++) {
            size_t gid = y * width + x;

            //3x3 sobel filter, first in x direction
            output_x[gid] = - in[idx(x, y, width, height, -1, -1)] +
                            in[idx(x, y, width, height, 1, -1)] -
                            2 * in[idx(x, y, width, height, -1, 0)] +
                            2 * in[idx(x, y, width, height, 1, 0)] -
                            in[idx(x, y, width, height, -1, 1)] +
                            in[idx(x, y, width, height, 1, 1)];

            //3x3 sobel filter, in y direction
            output_y[gid] = - in[idx(x, y, width, height, -1, -1)] +
                            in[idx(x, y, width, height, -1, 1)] -
                            2 * in[idx(x, y, width, height, 0, -1)] +
                            2 * in[idx(x, y, width, height, 0, 1)] -
                            in[idx(x, y, width, height, 1, -1)] +
                            in[idx(x, y, width, height, 1, 1)];
        }
    }
}
