__kernel void phaseAndMagnitude(global short *restrict in_x, global short *restrict in_y,
                                global uchar *restrict phase_out, global ushort *restrict magnitude_out) {

    uint id = get_global_id(0);

    ushort width = get_global_size(0);
    ushort height = get_global_size(0);

    short y=in_y[id];
    short x=in_x[id];

    float angle = atan2((float)y, (float)x);
    angle *= 40.5845104884;
    angle += 128;
    phase_out[id] = (ushort)angle;

    magnitude_out[id] = abs(x) + abs(y);
}
