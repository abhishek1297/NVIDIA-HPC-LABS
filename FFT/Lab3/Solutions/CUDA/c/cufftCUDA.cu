extern "C" {
    #include "../../../Common/c/fileReader.h"
    #include "../../../Common/c/cuhelper.h"
}
#include <stdio.h>
#include <math.h>
#include <unistd.h>
//CUDA includes
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Size of array in bytes
extern size_t MEM_SIZE;

//Device allocated pointers
extern Complex *d_inputFFT, *d_outputFFT;

//Host allocated pointer
extern Complex *h_errors;
extern int *d_freqArray;

//Timing variables
extern cudaEvent_t start, stop;
float execTime;



extern "C" __global__ void initSineWave(Complex *d_inputFFT, int *d_freqArray) {

    /**
    Each thread is executing this kernel and will compute and assign a random wave input
    for a given global thread index.
    */

    //global thread index
    int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tIdx < SIGNAL_SIZE) {
        double val = 0.0;
        double commonCalc = TWO_PI * tIdx / NUM_SAMPLES;
        double W1 = FREQUENCY * commonCalc;
        int j;
        for (j=0; j<ITERS; ++j) {
            
                double W2  = d_freqArray[j] * commonCalc;
                double DELTA_W = fabs(W1 - W2);
                double AVG_W = (W1 + W2) * 0.5;
                val += 2 * cos(DELTA_W * 0.5 * tIdx) * sin(AVG_W * tIdx);
        }
        d_inputFFT[tIdx].x = val / ITERS;
        d_inputFFT[tIdx].y = 0.0;
    }
}

void initialize() {
    /*
    * Intializing the complex number
    * with a random wave using a CUDA kernel.
    */
    int freqArray[ITERS];
    float time = 0.0;
    int GRID_DIM = ceil(SIGNAL_SIZE/(double)1024);
    int BLK_DIM = 1024;
    
    //Loading the frequencies from a file.
    loadFrequencies("../../../Common/frequencies.txt", freqArray, ITERS);
    cudaMemcpy(d_freqArray, freqArray, sizeof(freqArray), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start, 0);
    
    //Launching the CUDA kernel
    initSineWave<<<GRID_DIM, BLK_DIM>>>(d_inputFFT, d_freqArray);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    execTime += time / 1000;
    printf("INIT:%.2f\t", time / 1000);
}

void executeFFT() {
    
    float time = 0.0;
    //Creating a plan for Complex to Complex operation in double precision.
    cufftHandle plan;

    cudaEventRecord(start, 0);
    
    cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_Z2Z, 1); // Z2Z is C2C with Double Precision
    // Perform FFT 
    cufftExecZ2Z(plan, (cufftDoubleComplex *)d_inputFFT,
                    (cufftDoubleComplex *)d_outputFFT,
                    CUFFT_FORWARD);
                    
    // Perform Inverse FFT
    cufftExecZ2Z(plan, (cufftDoubleComplex *)d_outputFFT,
        (cufftDoubleComplex *)d_outputFFT,
        CUFFT_INVERSE);
    
    cufftDestroy(plan);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    execTime += time / 1000;
    printf("FFT:%.2f\t", time / 1000);
}

extern "C" __global__ void computeError(Complex *d_inputFFT,
                            Complex *d_outputFFT) {


    //global index
    int tIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (tIdx < SIGNAL_SIZE) {
        //Compute the difference
        //Normalize the output of the Inverse FFT by dividing it by SIZE beforehand
        //Store the differences in the output array
        d_outputFFT[tIdx].x = fabs(d_inputFFT[tIdx].x - (d_outputFFT[tIdx].x / SIGNAL_SIZE));
    }
}

void calculateDifference() {

    float time = 0.0;
    int GRID_DIM = ceil(SIGNAL_SIZE/(double)64);
    int BLK_DIM = 64;

    cudaEventRecord(start, 0);
    
    //Launching the CUDA kernel
    computeError<<<GRID_DIM, BLK_DIM>>>(d_inputFFT, d_outputFFT);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    execTime += time / 1000;
    printf("DIFF:%.2f\t", time / 1000);
}


int main(int argc, char **argv) {

    // Number of times to execute the parallel portions of the code
    // To run ./fftacc.out <N> or make ARGS="<N>"
    //By default the N = 1
    int i, k, N = 1;
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    printf("\n\n==============Execution================\n\n");
    printf("SIGNAL SIZE = %d\n\n", SIGNAL_SIZE);

    //Allocating pointers
    allocate();

    printf("Times in seconds\n\n");
    
    for (i=0; i<N; ++i) {
    
        //Intializing
        initialize();
        //Executing FFT + IFFT
        executeFFT();
        //Calculating difference between inputFFT and inverse outputFFT
        calculateDifference();
        printf("TOTAL:%.2f\n\n", execTime);
	sleep(10);
    }
    
    //Copy the errors to the CPU printMeanError uses these values
    cudaMemcpy(h_errors, d_outputFFT, MEM_SIZE, cudaMemcpyDeviceToHost);
    
    //printMeanError is a CPU function.
    printMeanError(h_errors);
    
    //Deallocating pointers
    deallocate();

    printf("\n\n==============Terminated================\n\n");
    return 0;
}
