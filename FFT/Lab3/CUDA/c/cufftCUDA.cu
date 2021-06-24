extern "C" {
    #include "../../Common/c/fileReader.h"
    #include "../../Common/c/cuhelper.h"
}
#include <stdio.h>
#include <math.h>
//CUDA includes
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Size of array in bytes
extern size_t MEM_SIZE;

/******************************
Complex type is nothing but,
typedef Complex double2

double2 is a CUDA special type to store 2 doubles

Complex data;
data.x; //real part
data.y; //imaginary part

Note that, Imaginary part is not used in this case.
******************************/

//Device allocated pointers
extern Complex *d_inputFFT, *d_outputFFT;

//Host allocated pointer
extern Complex *h_errors;
extern int *d_freqArray;



extern "C" __global__ void initSineWave(Complex *d_inputFFT, int *d_freqArray) {


    /*********************************************************************************
    * The following code is a sequential code that needs to be written inside this kernel.
    * The outer loop should be parallelized.
    * Everything inside the loop (////seq////) is executed sequentially by each thread.
    * Each occurance of i should be replace with the thread index
    *********************************************************************************/
    /*
    int i, j;
    double val, commonCalc, W1, W2, DELTA_W, AVG_W;
    for (i=0; i<SIGNAL_SIZE; ++i) { <== Parallelize this

        //////////////////Sequential Part//////////////////////
        
        val = 0.0;
        commonCalc = TWO_PI * i / NUM_SAMPLES;
        W1 = FREQUENCY * commonCalc;
        for (j=0; j<ITERS; ++j) {
            
                W2  = d_freqArray[j] * commonCalc;
                DELTA_W = fabs(W1 - W2);
                AVG_W = (W1 + W2) * 0.5;
                val += 2 * cos(DELTA_W * 0.5 * i) * sin(AVG_W * i);
        }
        d_inputFFT[i].x = val / ITERS;
        d_inputFFT[i].y = 0.0;
        
        //////////////////Sequential End//////////////////////
    }*/
}

void initialize() {
    /*
    * Intializing the complex number
    * with a random wave using a CUDA kernel.
    */
    /************************************************************
    * GRID dimension and BLOCK dimension are kernel launch parameters
    * Set those values in the following variables and then use them while
      launching the kernel.
    ************************************************************/
    int GRID_DIM = 0;
    int BLK_DIM = 0;

    int freqArray[ITERS];

    //Loading the frequencies from a file.
    loadFrequencies("../../Common/frequencies.txt", freqArray, ITERS);

    /***********************************************************************
    * Transfer the frequencies to the d_freqArray in the GPU.
    * Call the initSineWave kernel with above launch parameters here.
    * Synchronize the kernel launch as well.
    ************************************************************************/
}

void executeFFT() {

    cufftHandle plan;
    /***********************************************************************
    Do the following steps
    
    1. Create a 1D plan of length SIGNAL_SIZE in double precision
    for Complex to Complex Transform.

    2. Execute FFT (time to frequency) for given input signal
        -Do an out of place transform
        i.e use two different arrays for input and output
    
    3. Execute Inverse FFT (frequency to time) for the given output of fft
        -Do an in-place transform
        i.e use the same array for input and output. Use output array
        -Input array must not be updated
    
    Make sure you type cast the pointers to cufftDoubleComplex
    ***********************************************************************/
    cufftDestroy(plan);
}

extern "C" __global__ void computeError(Complex *d_inputFFT,
    Complex *d_outputFFT) {

    /***********************************************************************
    * Each thread is calculating the difference between InputFFT and outputFFT
    * Normalize outputFFT by the SIGNAL_SIZE
    * fabs() is used for finding the absolute value for a double precision number
    * Store the results in the real part of the output array
    * d_outputFFT[i].x = fabs(d_inputFFT[i].x - (d_outputFFT[i].x / SIGNAL_SIZE))
    * Due to memory limitations we had to overwrite the output array.
    ************************************************************************/
}

void calculateDifference() {
    
    /************************************************************
    * GRID dimension and BLOCK dimension are kernel launch parameters
    * Set those values in the following variables and then use them while
      launching the kernel.
    ************************************************************/
    int GRID_DIM = 0;
    int BLK_DIM = 0;
    /************************************
    * Launch the computeError kernel here
    * Synchronize the kernel launch as well.
    *************************************/
}

int main(int argc, char **argv) {

    printf("\n\n==============Execution================\n\n");
    printf("SIGNAL SIZE = %d\n\n", SIGNAL_SIZE);

    //Allocating pointers
    allocate();
    
    //Intializing
    initialize();
    //Executing FFT + IFFT
    executeFFT();
    //Calculating difference between inputFFT and inverse outputFFT
    calculateDifference();
    
    /************************************************************************
    * copy the difference results stored in h_outputFFT into h_errors on the CPU.
    * h_errors is required by the printMeanError()
    ************************************************************************/
       
    //printMeanError is a CPU function that calculates average errors from the differences
    printMeanError(h_errors);
    
    //Deallocating pointers
    deallocate();

    printf("\n\n==============Terminated================\n\n");
    return 0;
}