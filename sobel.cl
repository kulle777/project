uint idx(ushort x, ushort y, ushort width, ushort height, ushort xoff, ushort yoff) {
    ushort resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    ushort resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

kernel void sobel3x3(global uchar *restrict in, ushort width, ushort height,
    global short *restrict output_x, global short *restrict output_y) {

    int i = get_global_id(0);
    int j = get_global_id(1);

    // Dont run on the edge
    //if(i == 0 | i == width) return;
    //if(j == 0 | j == height)  return;

    uint gid = j*width + i;


    //3x3 sobel filter, first in x direction
    output_x[gid] = - in[idx(i, j, width, height, -1, -1)] +
                    in[idx(i, j, width, height, 1, -1)] -
                    2 * in[idx(i, j, width, height, -1, 0)] +
                    2 * in[idx(i, j, width, height, 1, 0)] -
                    in[idx(i, j, width, height, -1, 1)] +
                    in[idx(i, j, width, height, 1, 1)];

    //3x3 sobel filter, in y direction
    output_y[gid] = - in[idx(i, j, width, height, -1, -1)] +
                    in[idx(i, j, width, height, -1, 1)] -
                    2 * in[idx(i, j, width, height, 0, -1)] +
                    2 * in[idx(i, j, width, height, 0, 1)] -
                    in[idx(i, j, width, height, 1, -1)] +
                    in[idx(i, j, width, height, 1, 1)];

}
