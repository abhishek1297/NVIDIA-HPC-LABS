#ifndef __HELPER__
#define __HELPER__

#include <stdio.h>
#include <math.h>
#include <stddef.h>
//CUDA includes
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define SIGNAL_SIZE 536870912
//#define SIGNAL_SIZE 67108864
#define ITERS 1000

//For calculating the sine wave
#define TWO_PI 2 * M_PI
#define NUM_SAMPLES 44100
#define FREQUENCY 100

//CUFFT specific data type double2 (double, double)
typedef double2 Complex;

//Device allocated pointers
Complex *d_inputFFT, *d_outputFFT;

//Host allocated pointer
Complex *h_errors;
int *d_freqArray;

//Timing variables
cudaEvent_t start, stop;

void allocate();

void printMeanError(Complex *);

void deallocate();

#endif
