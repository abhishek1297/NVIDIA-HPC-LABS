#include "../../Inputs/c/fileReader.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
// CUDA includes
#include <cufft.h>

//Input Size
#define SIZE 134217728

typedef double2 Complex;

//Host allocated pointers
Complex *h_inputFFT, *h_outputFFT, *h_outputInvFFT;

//Timing variables
struct timeval start, stop;
float execTime;

void allocate() {

    h_inputFFT = (Complex *) malloc(sizeof(Complex) * SIZE);
    h_outputFFT = (Complex *) malloc(sizeof(Complex) * SIZE);
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
}

void executeFFT() {

    int i;
    double error = 0.0;
    //Creating a plan for CUFFT
    cufftHandle plan;
    
    gettimeofday(&start, NULL);

    cufftPlan1d(&plan, SIZE, CUFFT_Z2Z, 1); // Z2Z is C2C with Double Precision
   
    #pragma acc host_data use_device(h_inputFFT, h_outputFFT, h_outputInvFFT)
    {
        //Performing FFT FORWARD
        cufftExecZ2Z(plan, (cufftDoubleComplex *) h_inputFFT,
                    (cufftDoubleComplex *) h_outputFFT,
                    CUFFT_FORWARD);
        //Performing FFT INVERSE
        cufftExecZ2Z(plan, (cufftDoubleComplex *) h_outputFFT,
                    (cufftDoubleComplex *) h_outputInvFFT,
                    CUFFT_INVERSE);
    }
    
    cufftDestroy(plan);

    //Transfering data back to the CPU for comparison
    #pragma acc update self(h_outputInvFFT[:SIZE]) wait
    
    gettimeofday(&stop, NULL);

    //Accumulating total error
    for (i=0; i<SIZE; ++i) {
        error += fabs(h_inputFFT[i].x - h_outputInvFFT[i].x/SIZE);
    }
    
    printf("\nMean Error = %e\n", error / SIZE);

    execTime = (float)(stop.tv_usec - start.tv_usec) / 1000000 + (float)(stop.tv_sec - start.tv_sec);
}

void deallocate() {

    free(h_inputFFT);
    free(h_outputFFT);
    free(h_outputInvFFT);
}

int main () {

    allocate();
    
    initialize();
    #pragma acc data copyin(h_inputFFT[:SIZE]), create(h_outputFFT[:SIZE], h_outputInvFFT[:SIZE])
    {
        executeFFT();
    }
    
    printf("\nOnly FFT:");
    printf("\nExecution Time: %.10f s\n\n", execTime);
    deallocate();

    return 0;
}
