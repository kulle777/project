uint idx(ushort x, ushort y, ushort width, ushort height, short xoff, short yoff) {
    ushort resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    ushort resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

kernel void sobel3x3(global uchar *restrict in, global short *restrict output_x, global short *restrict output_y) {

    ushort x = get_global_id(0);
    ushort y = get_global_id(1);

    ushort width = get_global_size(0);
    ushort height = get_global_size(0);

    uint gid = y*width + x;

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
