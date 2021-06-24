#include "../../Inputs/c/fileReader.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
// CUDA includes
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Input Size
#define SIZE 134217728

typedef double2 Complex; // CUFFT special type for complex data

//Device allocated pointers
Complex *d_inputFFT, *d_outputFFT, *d_outputInvFFT;

//Host allocated pointers for printing
Complex *h_inputFFT, *h_outputInvFFT;

//Timing Variables
cudaEvent_t start, stop;
float execTime = 0.0;


void allocate() {

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc((void **)&d_inputFFT, sizeof(Complex) * SIZE);
    cudaMalloc((void **)&d_outputFFT, sizeof(Complex) * SIZE);
    cudaMalloc((void **)&d_outputInvFFT, sizeof(Complex) * SIZE);

    h_inputFFT = (Complex *) malloc(sizeof(Complex) * SIZE);
    h_outputInvFFT = (Complex *) malloc(sizeof(Complex) * SIZE);
}


void initialize() {

    int i;
    double inputs[SIZE];

    //Loading real numbers from a fle
    char *fname = "../../Inputs/complexInputs.txt";
    readInputs(fname, inputs, SIZE);

    for (i=0; i<SIZE; ++i) {
        h_inputFFT[i].x = inputs[i];
        h_inputFFT[i].y = 0.0;
    }

    cudaMemcpy(d_inputFFT, h_inputFFT, sizeof(Complex) * SIZE, cudaMemcpyHostToDevice);
}

void executeFFT() {

    int i;
    double error = 0.0;
    //Creating a plan for CUFFT
    cufftHandle plan;
    
    cudaEventRecord(start, 0);

    cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1); // Z2Z is C2C with Double Precision

    //Performing FFT FORWARD
    cufftExecZ2Z(plan, (cufftDoubleComplex *) d_inputFFT,
                    (cufftDoubleComplex *) d_outputFFT,
                    CUFFT_FORWARD);

    //Performing FFT INVERSE
    cufftExecZ2Z(plan, (cufftDoubleComplex *) d_outputFFT,
        (cufftDoubleComplex *) d_outputInvFFT,
        CUFFT_INVERSE);

    cufftDestroy(plan);

    //Transfering data back to the CPU for comparison
    cudaMemcpy(h_outputInvFFT, d_outputInvFFT, sizeof(Complex) * SIZE, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //Accumulating total error
    for (i=0; i<SIZE; ++i) {
        error += fabs(h_inputFFT[i].x - h_outputInvFFT[i].x/SIZE);
    }

    printf("\nMean Error = %e\n", error / SIZE);

    cudaEventElapsedTime(&execTime, start, stop);
}

void deallocate() {

    cudaFree(d_inputFFT);
    cudaFree(d_outputFFT);
    cudaFree(d_outputInvFFT);
    
    free(h_inputFFT);
    free(h_outputInvFFT);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {

    allocate();
    
    initialize();
    
    executeFFT();
    
    printf("\nOnly FFT:");
    printf("\nExecution Time =  %.10f s\n", execTime / 1000);
    
    deallocate();
    
    return 0;
}
