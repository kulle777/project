/* COMP.CE.350 Parallelization Excercise 2023
   Copyright (c) 2023 Topi Leppanen topi.leppanen@tuni.fi
                      Jan Solanti

VERSION 23.0 - Created

Opencl implementation by Kalle Paasio
*/


#define CL_TARGET_OPENCL_VERSION 300


#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <CL/cl.h>

#include "util.h"
#include "opencl_util.h"

// Globals by kalle
cl_platform_id *g_platforms;
cl_uint g_numDevices;
cl_device_id *g_devices;
cl_context g_context;
cl_command_queue g_cmdQueue;

cl_mem g_buf_sobel_in;
cl_mem g_buf_sobel_out_x;
cl_mem g_buf_sobel_out_y;
cl_mem g_buf_phase_out;
cl_mem g_buf_magnitude_out;
cl_mem g_buf_nonmax_out;

cl_kernel g_sobel_kernel;
cl_kernel g_phase_kernel;
cl_kernel g_nonmax_kernel;

size_t g_image_size;


// Is used to find out frame times
int previousFinishTime = 0;
unsigned int frameNumber = 0;
unsigned int seed = 0;

typedef struct {
    uint16_t x;
    uint16_t y;
} coord_t;

const coord_t neighbour_offsets[8] = {
    {-1, -1}, {0, -1},  {+1, -1}, {-1, 0},
    {+1, 0},  {-1, +1}, {0, +1},  {+1, +1},
};


// Jos voisin kaantaa logiikan ja laskea olenko MINA jonkun neighbour ja pitaisiko oma arvo muuttaa
// Tuolloin parallelisoituisi. Sitten vaan laske KAIKKI... ongelma edelleen etta propagaatio ei ehka ole tapahtunut viela ja olisi pitanytkin menna YES pikseliksi.
// Pitkaa laskea aina uudestaan kun tulee uusi yes pikseli.
//HEIHEI ongelmaa datan kanssa EI OLE. Me voidaan aina kirjoittaa YES päälle. Sitten rekursiolla pitää vaan ajaa kyseinen jäbä uudelleen. Koskaan ei kirjoiteta 0 ajon aikana, joten ei voi kusta.

// Toinen vaihtoehto: rekursio

void
edgeTracing(uint8_t *restrict image, size_t width, size_t height) {
    // Uses a stack-based approach to incrementally spread the YES
    // pixels to every (8) neighbouring MAYBE pixel.
    //
    // Modifies the pixels in-place.
    //
    // Since the same pixel is never added to the stack twice,
    // the maximum stack size is quaranteed to never be above
    // the image size and stack overflow should be impossible
    // as long as stack size is 2*2*image_size (2 16-bit coordinates per
    // pixel).
    coord_t *tracing_stack = malloc(width * height * sizeof(coord_t));
    coord_t *tracing_stack_pointer = tracing_stack;


    // LOOP 4.1
    for (uint16_t y = 0; y < height; y++) {
        // LOOP 4.2
        for (uint16_t x = 0; x < width; x++) {
            // Collect all YES pixels into the stack
            if (image[y * width + x] == 255) {
                coord_t yes_pixel = {x, y};
                *tracing_stack_pointer = yes_pixel;
                tracing_stack_pointer++;
            }
        }
    }


    // Empty the tracing stack one-by-one
    // LOOP 4.3
    while (tracing_stack_pointer != tracing_stack) {
        tracing_stack_pointer--;
        coord_t known_edge = *tracing_stack_pointer;

        // LOOP 4.4
        for (int k = 0; k < 8; k++) {
            coord_t dir_offs = neighbour_offsets[k];
            coord_t neighbour = {known_edge.x + dir_offs.x, known_edge.y + dir_offs.y};

            // Clamp to edge to prevent the algorithm from leaving the image.
            // Not using the idx()-function, since we want to preserve the x
            // and y on their own, since the pixel might be added to the stack
            // in the end.
            if (neighbour.x < 0) neighbour.x = 0;
            if (neighbour.x >= width) neighbour.x = width - 1;
            if (neighbour.y < 0) neighbour.y = 0;
            if (neighbour.y >= height) neighbour.y = height - 1;

            // Only MAYBE neighbours are potential edges
            if (image[neighbour.y * width + neighbour.x] == 127) {
                // Convert MAYBE to YES
                image[neighbour.y * width + neighbour.x] = 255;

                // Add the newly added pixel to stack, so changes will
                // propagate
                *tracing_stack_pointer = neighbour;
                tracing_stack_pointer++;
            }
        }
    }
    // Clear all remaining MAYBE pixels to NO, these were not reachable from
    // any YES pixels
    // LOOP 4.5
    #pragma omp parallel for
    for (size_t n = 0; n < width*height; n++){
        if (image[n] == 127) image[n] = 0;
    }
    free(tracing_stack);
}


void cl_sobel(size_t width, size_t height, cl_event* event){
    // height and width determined as 16 bits -> max image size is 65 535 x 65 535 pixels
    cl_int status;

    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(g_sobel_kernel, 0, sizeof(cl_mem), &g_buf_sobel_in);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_sobel_kernel, 1, sizeof(cl_mem), &g_buf_sobel_out_x);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_sobel_kernel, 2, sizeof(cl_mem), &g_buf_sobel_out_y);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}

    size_t globalWorkSize[2] = {width, height};     // 100x100 for x.pgm, 4444x4395 for hameensilta.pgm

    status = clEnqueueNDRangeKernel(g_cmdQueue, g_sobel_kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, event);
    if(status != CL_SUCCESS){printf("Error: Enque kernel. Err no %d\n",status);}
}


void cl_phase(size_t width, size_t height, cl_event* event){
    cl_int status;
    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(g_phase_kernel, 0, sizeof(cl_mem), &g_buf_sobel_out_x);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_phase_kernel, 1, sizeof(cl_mem), &g_buf_sobel_out_y);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_phase_kernel, 2, sizeof(cl_mem), &g_buf_phase_out);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_phase_kernel, 3, sizeof(cl_mem), &g_buf_magnitude_out);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}

    size_t globalWorkSize[1] = {width*height};

    status = clEnqueueNDRangeKernel(g_cmdQueue, g_phase_kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, event);
    if(status != CL_SUCCESS){printf("Error: Enque kernel. Err no %d\n",status);}
}


void cl_nonmax(size_t width, size_t height, uint16_t threshold_lower,uint16_t threshold_upper, cl_event* event){
    cl_int status;
    cl_ushort low = threshold_lower;
    cl_ushort high = threshold_upper;
    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(g_nonmax_kernel, 0, sizeof(cl_mem), &g_buf_magnitude_out);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_nonmax_kernel, 1, sizeof(cl_mem), &g_buf_phase_out);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_nonmax_kernel, 2, sizeof(cl_ushort), &low);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_nonmax_kernel, 3, sizeof(cl_ushort), &high);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}
    status = clSetKernelArg(g_nonmax_kernel, 4, sizeof(cl_mem), &g_buf_nonmax_out);
    if(status != CL_SUCCESS){printf("Error: kernel arg. Err no %d\n",status);}

    size_t globalWorkSize[2] = {width,height};

    status = clEnqueueNDRangeKernel(g_cmdQueue, g_nonmax_kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, event);
    if(status != CL_SUCCESS){printf("Error: Enque kernel. Err no %d\n",status);}   
}


// This is the main processing element of the code which sends kernels to the device. Moves data and captures timings
void cannyEdgeDetection(
    uint8_t *restrict input, size_t width, size_t height,
    uint16_t threshold_lower, uint16_t threshold_upper,
    uint8_t *restrict output, double *restrict runtimes) {

    uint64_t times[3];
    times[0] = gettimemono_ns();
    cl_int status;
    cl_event write_buf_event, read_buf_event, sobel_event, phase_event, nonmax_event;

    // Send the data to the GPU
    status = clEnqueueWriteBuffer(g_cmdQueue, g_buf_sobel_in, CL_FALSE, 0, g_image_size*sizeof(uint8_t), input, 0, NULL, &write_buf_event);
    if(status != CL_SUCCESS){printf("Error: Enque buffer. Err no %d\n",status);}

    cl_sobel(width, height, &sobel_event);

    cl_phase(width, height, &phase_event);

    cl_nonmax(width, height, threshold_lower, threshold_upper, &nonmax_event);

    // Read data back to CPU
    status = clEnqueueReadBuffer(g_cmdQueue, g_buf_nonmax_out, CL_FALSE, 0, g_image_size*sizeof(uint8_t), output, 0, NULL, &read_buf_event);
    if(status != CL_SUCCESS){printf("Error: Enque buffer. Err no %d\n",status);}
    // You MUST wait for all the data to come back, as CPU can't deal with incomplete data
    clFinish(g_cmdQueue);

    times[1] = gettimemono_ns();
    edgeTracing(output, width, height);  // modifies output in-place
    times[2] = gettimemono_ns();

    printf("Whole time of cannyEdgeDetection including delays and buffers %.2f ms\n", (times[2] - times[0])/1000000.);
    printf("Time for clEnqueueWriteBuffer: %.2f ms\n", getStartEndTime(write_buf_event)/1000000.);
    printf("Time for clEnqueueReadBuffer: %.2f ms\n", getStartEndTime(read_buf_event)/1000000.);

    runtimes[0] = getStartEndTime(sobel_event)/1000000.;
    runtimes[1] = getStartEndTime(phase_event)/1000000.;
    runtimes[2] = getStartEndTime(nonmax_event)/1000000.;
    runtimes[3] = (times[2] - times[1])/1000000.;
}


// Needed only in Part 2 for OpenCL initialization
void
init(
    size_t width, size_t height, uint16_t threshold_lower,
    uint16_t threshold_upper) {

    g_image_size = width*height;

    // Use this to check the output of each API call
    cl_int status;

    // Retrieve the number of platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    // Allocate enough space for each platform
    g_platforms = (cl_platform_id*)malloc(
        numPlatforms*sizeof(cl_platform_id));

    // Fill in the platforms
    status = clGetPlatformIDs(numPlatforms, g_platforms, NULL);
    if(status != CL_SUCCESS){printf("Error: clGetPlatformIDs. Err no %d\n",status);}

    // Retrieve the number of devices
    status = clGetDeviceIDs(g_platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &g_numDevices);
    if(status != CL_SUCCESS){printf("Error: clGetDeviceIDs. Err no %d\n",status);}

    // Allocate enough space for each device
    g_devices = (cl_device_id*)malloc(g_numDevices*sizeof(cl_device_id));

    // Fill in the devices
    status = clGetDeviceIDs(g_platforms[0], CL_DEVICE_TYPE_ALL, g_numDevices, g_devices, NULL);
    if(status != CL_SUCCESS){printf("Error: clGetDeviceIDs2. Err no %d\n",status);}

    // Create a context and associate it with the devices
    g_context = clCreateContext(NULL, g_numDevices, g_devices, NULL, NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateContext. Err no %d\n",status);}

    // Create a command queue and associate it with the device
    cl_queue_properties proprt[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    g_cmdQueue = clCreateCommandQueueWithProperties(g_context, g_devices[0], proprt, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateCommandQueue. Err no %d\n",status);}

    //buffers
    g_buf_sobel_in = clCreateBuffer(g_context, CL_MEM_READ_ONLY, g_image_size*sizeof(uint8_t), NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateBuffer g_buf_sobel_in. Err no %d\n",status);}

    g_buf_sobel_out_x = clCreateBuffer(g_context, CL_MEM_READ_WRITE, g_image_size*sizeof(int16_t), NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateBuffer g_buf_sobel_out_x. Err no %d\n",status);}
    g_buf_sobel_out_y = clCreateBuffer(g_context, CL_MEM_READ_WRITE, g_image_size*sizeof(int16_t), NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateBuffer g_buf_sobel_out_y. Err no %d\n",status);}

    g_buf_phase_out = clCreateBuffer(g_context, CL_MEM_READ_WRITE, g_image_size*sizeof(uint8_t), NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateBuffer g_buf_sobel_out_x. Err no %d\n",status);}
    g_buf_magnitude_out = clCreateBuffer(g_context, CL_MEM_READ_WRITE, g_image_size*sizeof(uint16_t), NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateBuffer g_buf_sobel_out_y. Err no %d\n",status);}

    g_buf_nonmax_out = clCreateBuffer(g_context, CL_MEM_WRITE_ONLY, g_image_size*sizeof(uint8_t), NULL, &status);
    if(status != CL_SUCCESS){printf("Error: clCreateBuffer g_buf_sobel_out_x. Err no %d\n",status);}

    size_t* max_work_group_size;
    size_t* max_work_items;

    max_work_group_size = malloc(1*sizeof(size_t));
    status = clGetDeviceInfo(g_devices[0],CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), max_work_group_size, NULL);
    if(status != CL_SUCCESS){printf("Error: get status. Err no %d\n",status);}
    printf("Max work group size %zu\n",max_work_group_size[0]);
    free(max_work_group_size);

    max_work_items = malloc(1*sizeof(size_t));
    status = clGetDeviceInfo(g_devices[0],CL_DEVICE_ADDRESS_BITS, sizeof(size_t), max_work_items, NULL);
    if(status != CL_SUCCESS){printf("Error: get status. Err no %d\n",status);}
    printf("Max work item size must be less than 2**%zu\n", max_work_items[0]);
    free(max_work_items);


    // Compile all the kernels beforehand

    ////////////////////////////////////////////////Sobel//////////////////////////////////////////////////////////////////
    // Create a program with source code
    char *sobel_source;
    sobel_source = read_source("sobel.cl");
    cl_program program = clCreateProgramWithSource(g_context, 1, (const char**)&sobel_source, NULL, &status);
    if(status != CL_SUCCESS){printf("Error: Create program. Errno %d\n",status);}
    status = clBuildProgram(program, g_numDevices, g_devices, NULL, NULL, NULL);
    if(status != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program,g_devices[0],CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program,g_devices[0],CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error: Build \n%s\n", log);
        free(log);
        return;
    }
    g_sobel_kernel = clCreateKernel(program, "sobel3x3", &status);
    if(status != CL_SUCCESS){printf("Error: Create kernel. Err no %d\n",status);}
    free(sobel_source);
    clReleaseProgram(program);

    ////////////////////////////////////////////////PhaseMagnitude//////////////////////////////////////////////////////////////////
    char *phase_source;
    phase_source = read_source("phase.cl");
    program = clCreateProgramWithSource(g_context, 1, (const char**)&phase_source, NULL, &status);
    if(status != CL_SUCCESS){printf("Error: Create program. Errno %d\n",status);}
    status = clBuildProgram(program, g_numDevices, g_devices,NULL, NULL, NULL);
    if(status != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program,g_devices[0],CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program,g_devices[0],CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error: Build \n%s\n", log);
        free(log);
        return;
    }
    g_phase_kernel = clCreateKernel(program, "phaseAndMagnitude", &status);
    if(status != CL_SUCCESS){printf("Error: Create kernel. Err no %d\n",status);}
    free(phase_source);
    clReleaseProgram(program);

    ////////////////////////////////////////////////Nonmax//////////////////////////////////////////////////////////////////
    char *nonmax_source;
    nonmax_source = read_source("nonmax.cl");
    program = clCreateProgramWithSource(g_context, 1, (const char**)&nonmax_source, NULL, &status);
    if(status != CL_SUCCESS){printf("Error: Create program. Errno %d\n",status);}
    status = clBuildProgram(program, g_numDevices, g_devices,"-Werror", NULL, NULL);
    if(status != CL_SUCCESS){
        size_t log_size;
        clGetProgramBuildInfo(program,g_devices[0],CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program,g_devices[0],CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Error: Build \n%s\n", log);
        free(log);
        return;
    }
    g_nonmax_kernel = clCreateKernel(program, "nonMaxSuppression", &status);
    if(status != CL_SUCCESS){printf("Error: Create kernel. Err no %d\n",status);}

    free(nonmax_source);
    clReleaseProgram(program);
}

void
destroy() {


    // Free OpenCL resources
    clReleaseKernel(g_nonmax_kernel);
    clReleaseKernel(g_phase_kernel);
    clReleaseKernel(g_sobel_kernel);

    clReleaseCommandQueue(g_cmdQueue);
    clReleaseMemObject(g_buf_sobel_in);
    clReleaseMemObject(g_buf_sobel_out_x);
    clReleaseMemObject(g_buf_sobel_out_y);
    clReleaseMemObject(g_buf_phase_out);
    clReleaseMemObject(g_buf_magnitude_out);
    clReleaseMemObject(g_buf_nonmax_out);
    clReleaseContext(g_context);

    free(g_platforms);
    free(g_devices);
}

////////////////////////////////////////////////
// ¤¤ DO NOT EDIT ANYTHING AFTER THIS LINE ¤¤ //
////////////////////////////////////////////////

enum PROCESSING_MODE { DEFAULT, BIG_MODE, SMALL_MODE, VIDEO_MODE };
// ¤¤ DO NOT EDIT THIS FUNCTION ¤¤
int
main(int argc, char **argv) {
    enum PROCESSING_MODE mode = DEFAULT;
    if (argc > 1) {
        char *mode_c = argv[1];
        if (strlen(mode_c) == 2) {
            if (strncmp(mode_c, "-B", 2) == 0) {
                mode = BIG_MODE;
            } else if (strncmp(mode_c, "-b", 2) == 0) {
                mode = SMALL_MODE;
            } else if (strncmp(mode_c, "-v", 2) == 0) {
                mode = VIDEO_MODE;
            } else {
                printf(
                    "Invalid usage! Please set either -b, -B, -v or "
                    "nothing\n");
                return -1;
            }
        } else {
            printf("Invalid usage! Please set either -b, -B, -v nothing\n");
            return -1;
        }
    }
    int benchmarking_iterations = 1;
    if (argc > 2) {
        benchmarking_iterations = atoi(argv[2]);
    }

    char *input_image_path = "";
    char *output_image_path = "";
    uint16_t threshold_lower = 0;
    uint16_t threshold_upper = 0;
    switch (mode) {
        case BIG_MODE:
            input_image_path = "hameensilta.pgm";
            output_image_path = "hameensilta_output.pgm";
            // Arbitrarily selected to produce a nice-looking image
            // DO NOT CHANGE THESE WHEN BENCHMARKING
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Enabling %d benchmarking iterations with the large %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case SMALL_MODE:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            threshold_lower = 750;
            threshold_upper = 800;
            printf(
                "Enabling %d benchmarking iterations with the small %s "
                "image\n",
                benchmarking_iterations, input_image_path);
            break;
        case VIDEO_MODE:
            if (system("which ffmpeg > /dev/null 2>&1") ||
                system("which ffplay > /dev/null 2>&1")) {
                printf(
                    "Video mode is disabled because ffmpeg is not found\n");
                return -1;
            }
            benchmarking_iterations = 0;
            input_image_path = "people.mp4";
            threshold_lower = 120;
            threshold_upper = 300;
            printf(
                "Playing video %s with FFMPEG. Error check disabled.\n",
                input_image_path);
            break;
        case DEFAULT:
        default:
            input_image_path = "x.pgm";
            output_image_path = "x_output.pgm";
            // Carefully selected to produce a discontinuous edge without edge
            // tracing
            threshold_lower = 750;
            threshold_upper = 800;
            printf("Running with %s image\n", input_image_path);
            break;
    }

    uint8_t *input_image = NULL;
    size_t width = 0;
    size_t height = 0;
    if (mode == VIDEO_MODE) {
        width = 3840;
        height = 2160;
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int count;
        uint8_t *frame = malloc(width * height * 3);
        assert(frame);
        char pipein_cmd[1024];
        snprintf(
            pipein_cmd, 1024,
            "ffmpeg -i %s -f image2pipe -vcodec rawvideo -an -s %zux%zu "
            "-pix_fmt gray - 2> /dev/null",
            input_image_path, width, height);
        FILE *pipein = popen(pipein_cmd, "r");
        char pipeout_cmd[1024];
        snprintf(
            pipeout_cmd, 1024,
            "ffplay -f rawvideo -pixel_format gray -video_size %zux%zu "
            "-an - 2> /dev/null",
            width, height);
        FILE *pipeout = popen(pipeout_cmd, "w");
        double runtimes[4];
        while (1) {
            count = fread(frame, 1, height * width, pipein);
            if (count != height * width) break;

            cannyEdgeDetection(
                frame, width, height, threshold_lower, threshold_upper,
                output_image, runtimes);

            double total_time =
                runtimes[0] + runtimes[1] + runtimes[2] + runtimes[3];
            printf("FPS: %0.1f\n", 1000 / total_time);
            fwrite(output_image, 1, height * width, pipeout);
        }
        fflush(pipein);
        pclose(pipein);
        fflush(pipeout);
        pclose(pipeout);
    } else {
        if ((input_image = read_pgm(input_image_path, &width, &height))) {
            printf(
                "Input image read succesfully. Size %zux%zu\n", width,
                height);
        } else {
            printf("Read failed\n");
            return -1;
        }
        init(width, height, threshold_lower, threshold_upper);

        uint8_t *output_image = malloc(width * height);
        assert(output_image);

        int all_the_runs_were_succesful = 1;
        double avg_runtimes[4] = {0.0, 0.0, 0.0, 0.0};
        double avg_total = 0.0;
        for (int iter = 0; iter < benchmarking_iterations; iter++) {
            double iter_runtimes[4];
            cannyEdgeDetection(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image, iter_runtimes);

            for (int n = 0; n < 4; n++) {
                avg_runtimes[n] += iter_runtimes[n] / benchmarking_iterations;
                avg_total += iter_runtimes[n] / benchmarking_iterations;
            }

            uint8_t *output_image_ref = malloc(width * height);
            assert(output_image_ref);
            cannyEdgeDetection_ref(
                input_image, width, height, threshold_lower, threshold_upper,
                output_image_ref);

            uint8_t *fused_comparison = malloc(width * height);
            assert(fused_comparison);
            int failed = validate_result(
                output_image, output_image_ref, width, height,
                fused_comparison);
            if (failed) {
                all_the_runs_were_succesful = 0;
                printf(
                    "Error checking failed for benchmark iteration %d!\n"
                    "Writing your output to %s. The image that should've "
                    "been generated is written to ref.pgm\n"
                    "Generating fused.pgm for debugging purpose. Light-grey "
                    "pixels should've been white and "
                    "dark-grey pixels black. Corrupted pixels are colored "
                    "middle-grey\n",
                    iter, output_image_path);

                write_pgm("ref.pgm", output_image_ref, width, height);
                write_pgm("fused.pgm", fused_comparison, width, height);
            }
        }

        printf("Sobel3x3 time          : %0.3f ms\n", avg_runtimes[0]);
        printf("phaseAndMagnitude time : %0.3f ms\n", avg_runtimes[1]);
        printf("nonMaxSuppression time : %0.3f ms\n", avg_runtimes[2]);
        printf("edgeTracing time       : %0.3f ms\n", avg_runtimes[3]);
        printf("Total time             : %0.3f ms\n", avg_total);
        write_pgm(output_image_path, output_image, width, height);
        printf("Wrote output to %s\n", output_image_path);
        if (all_the_runs_were_succesful) {
            printf("Error checks passed!\n");
        } else {
            printf("There were failing runs\n");
        }
    }
    destroy();
    return 0;
}
