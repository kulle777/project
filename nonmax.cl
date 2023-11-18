uint idx(ushort x, ushort y, ushort width, ushort height, short xoff, short yoff) {
    ushort resx = x;
    if ((xoff > 0 && x < width - xoff) || (xoff < 0 && x >= (-xoff)))
        resx += xoff;
    ushort resy = y;
    if ((yoff > 0 && y < height - yoff) || (yoff < 0 && y >= (-yoff)))
        resy += yoff;
    return resy * width + resx;
}

__kernel
void nonMaxSuppression(global ushort *restrict magnitude, global uchar *restrict phase,
         global ushort* threshold_lower, global ushort* threshold_upper, global uchar *restrict out) {

    ushort x = get_global_id(0);
    ushort y = get_global_id(1);

    ushort width = get_global_size(0);
    ushort height = get_global_size(0);

    uint gid = y * width + x;

    uchar sobel_angle = phase[gid];

    if (sobel_angle > 127) {
        sobel_angle -= 128;
    }

    ushort sobel_orientation = 0;

    if (sobel_angle < 16 || sobel_angle >= (7 * 16)) {
        sobel_orientation = 2;
    } else if (sobel_angle >= 16 && sobel_angle < 16 * 3) {
        sobel_orientation = 1;
    } else if (sobel_angle >= 16 * 3 && sobel_angle < 16 * 5) {
        sobel_orientation = 0;
    } else if (sobel_angle > 16 * 5 && sobel_angle <= 16 * 7) {
        sobel_orientation = 3;
    }

    ushort sobel_magnitude = magnitude[gid];
    /* Non-maximum suppression
     * Pick out the two neighbours that are perpendicular to the
     * current edge pixel */
    ushort neighbour_max = 0;
    ushort neighbour_max2 = 0;
    switch (sobel_orientation) {
        case 0:
            neighbour_max =
                magnitude[idx(x, y, width, height, 0, -1)];
            neighbour_max2 =
                magnitude[idx(x, y, width, height, 0, 1)];
            break;
        case 1:
            neighbour_max =
                magnitude[idx(x, y, width, height, -1, -1)];
            neighbour_max2 =
                magnitude[idx(x, y, width, height, 1, 1)];
            break;
        case 2:
            neighbour_max =
                magnitude[idx(x, y, width, height, -1, 0)];
            neighbour_max2 =
                magnitude[idx(x, y, width, height, 1, 0)];
            break;
        case 3:
        default:
            neighbour_max =
                magnitude[idx(x, y, width, height, 1, -1)];
            neighbour_max2 =
                magnitude[idx(x, y, width, height, -1, 1)];
            break;
    }
    // Suppress the pixel here
    if ((sobel_magnitude < neighbour_max) ||
        (sobel_magnitude < neighbour_max2)) {
        sobel_magnitude = 0;
    }

    /* Double thresholding */
    // Marks YES pixels with 255, NO pixels with 0 and MAYBE pixels
    // with 127
    uchar t = 127;
    if (sobel_magnitude > *threshold_upper){ t = 255;}
    if (sobel_magnitude <= *threshold_lower){ t = 0;}
    out[gid] = t;

}
