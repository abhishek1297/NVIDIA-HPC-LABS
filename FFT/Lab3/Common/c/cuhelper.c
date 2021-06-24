#include "cuhelper.h"

size_t MEM_SIZE = sizeof(Complex) * SIGNAL_SIZE;

void allocate() {

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_inputFFT, MEM_SIZE);
    cudaMalloc((void **)&d_outputFFT, MEM_SIZE);
    cudaMalloc((void **)&d_freqArray, sizeof(int) * ITERS);
    h_errors = (Complex *) malloc(MEM_SIZE);    
}

void printMeanError(Complex *h_errors) {

    double errorSum = 0.0;
    int i;
    //Adding the differences
    for (i=0; i<SIGNAL_SIZE; ++i)
        errorSum += h_errors[i].x;
    //Show the Mean Error
    printf("\n\nMean Error = %e", errorSum / SIGNAL_SIZE);
}

void deallocate() {

    cudaFree(d_inputFFT);
    cudaFree(d_outputFFT);
    cudaFree(d_freqArray);
    free(h_errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}