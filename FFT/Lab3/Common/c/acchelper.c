#include "acchelper.h"

size_t MEM_SIZE = sizeof(Complex) * SIGNAL_SIZE;

void allocate() {
 
    h_inputFFT = (Complex *)malloc(MEM_SIZE);
    h_outputFFT = (Complex *)malloc(MEM_SIZE);
}

void printMeanError(Complex *h_errors) {
    
    int i;
    double errorSum = 0.0;
    //Adding the differences
    for (i=0; i<SIGNAL_SIZE; ++i)
        errorSum += h_errors[i].x;
    //Show the Mean Error
    printf("\n\nMean Error =  %e", errorSum / SIGNAL_SIZE);
}

void deallocate() {
 
    free(h_inputFFT);
    free(h_outputFFT);
}