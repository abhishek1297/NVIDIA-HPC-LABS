#ifndef __ACCHELPER__
#define __ACCHELPER__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
//CUDA includes
#include <cufft.h>

#define SIGNAL_SIZE 536870912
//#define SIGNAL_SIZE 67108864
#define ITERS 1000

//For calculating the signal wave
#define TWO_PI 2 * M_PI
#define NUM_SAMPLES 44100
#define FREQUENCY 100

//CUFFT specific data type double2 (double, double)
typedef double2 Complex;

//Host allocated pointers
Complex *h_inputFFT, *h_outputFFT;

//Timing variables
struct timeval start, stop;

void allocate();

void printMeanError(Complex *);

void deallocate();

#endif
