__kernel
void vecadd(__global int *A,
            __global int *B,
            __global int *C)
{

   // Get the work-item’s unique ID
   int idx = get_global_id(0);

   // Add the corresponding locations of
   // 'A' and 'B', and store the result in 'C'.
   C[idx] = A[idx] + B[idx];
}

__kernel
void
phaseAndMagnitude(
    const int16_t *restrict in_x, const int16_t *restrict in_y, size_t width,
    size_t height, uint8_t *restrict phase_out,
    uint16_t *restrict magnitude_out) {
    // LOOP 2.1

    // we COULD separate phase and magnitude, but it is not fast
    // Unroll indexes and pre calculate weights
    for (size_t id = 0; id<width*height;id++){
        // Do memory operations serially before moving to split phases
        int16_t y=in_y[id];
        int16_t x=in_x[id];

        float angle = atan2f(y, x);
        angle *= 40.5845104884;
        angle += 128;
        phase_out[id] = (uint8_t)angle;

        magnitude_out[id] = abs(x) + abs(y);
    }
}
