#include "../../Inputs/c/fileReader.h"
#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

// Input Size
#define SIZE 134217728
//#define SIZE 67108864
typedef fftw_complex Complex;

//Host allocated pointers
Complex *inputFFT, *outputFFT;

//Timing Variables
struct timeval start, stop;
double execTime = 0.0;

void allocate() {

    inputFFT = (Complex *) malloc(sizeof(Complex) * SIZE);// data[N][2]
    outputFFT = (Complex *) malloc(sizeof(Complex) * SIZE);
}

void initialize() {

    int i;
    double inputs[SIZE];

    printf("\nLoading file...\n");

    //Loading real numbers from a fle
    readInputs("../../Inputs/complexInputs.txt", inputs, SIZE);

    for (i=0; i<SIZE; ++i) {
        inputFFT[i][0] = inputs[i];
        inputFFT[i][1] = 0.0;
    }
}

void executeFFT() {

    int i;
    double error = 0.0;
    
    //Creating a plan for FFT FORWARD
    fftw_plan plan = fftw_plan_dft_1d(SIZE, inputFFT, outputFFT, FFTW_FORWARD, FFTW_ESTIMATE);
    
    //Performing FFT FORWARD
    fftw_execute(plan);
    
    //Creating a plan for FFT INVERSE
    plan = fftw_plan_dft_1d(SIZE, outputFFT, outputFFT, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    //Performing FFT INVERSE
    fftw_execute(plan);
    fftw_destroy_plan(plan);
    
    //Accumulating total error
    for  (i=0; i<SIZE; ++i) {
        error += fabs(inputFFT[i][0] - outputFFT[i][0]/SIZE);
    }

    printf("\nMean Error = %e\n", error / SIZE);
}

void multicoreFFT() {

    int threads = omp_get_max_threads();
    printf("\n***Total CPU THREADS = %d\n", threads);
    
    if (fftw_init_threads() == 0) {
        printf("\nError Initializing Threads.\n");
        exit(1);
    }

    fftw_plan_with_nthreads(threads);
    fftw_make_planner_thread_safe();
    
    //Now the following FFT call will be planned with multi-threading
    executeFFT();
    
    fftw_cleanup_threads();
}


void deallocate() {

    free(inputFFT);
    free(outputFFT);
}

int main(int argc, char **argv) {

    allocate();

    printf("\nInitializing...\n");
    initialize();
    
    gettimeofday(&start, NULL);
    
    if (argc > 1 && strcmp(argv[1], "-m") == 0) {
        printf("\nMulticore Execution\n");
        multicoreFFT();
    }
    else {
        printf("\nUnicore Execution\n");
        executeFFT();
    }

    gettimeofday(&stop, NULL);

    execTime = (double)(stop.tv_usec - start.tv_usec) / 1000000 + (double)(stop.tv_sec - start.tv_sec);
    
    printf("\nExecution Time(sec) =  %.10f\n", execTime);
    
    deallocate();
    return 0;
}

