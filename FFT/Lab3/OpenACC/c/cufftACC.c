#include "../../Common/c/fileReader.h"
#include "../../Common/c/acchelper.h"
#include <stdio.h>
#include <math.h>
//CUDA includes
#include <cufft.h>

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

//Size of array in bytes
extern size_t MEM_SIZE;

//Host allocated pointers
extern Complex *h_inputFFT, *h_outputFFT;



void initialize() {

    /**
     * Initializing the complex number
     * with a random wave using Openacc directives
     */
    
    int i, j;
    double val, commonCalc, W1, W2, DELTA_W, AVG_W;
    int h_freqArray[ITERS];

    //Loading the frequencies from a file.
    loadFrequencies("../../Common/frequencies.txt", h_freqArray, ITERS);
    
    /**************************************************************
    * Parallelize the following loop with appropriate data clauses
    ***************************************************************/
    for (i=0; i<SIGNAL_SIZE; ++i) {
        val = 0.0;
        commonCalc = TWO_PI * i / NUM_SAMPLES;
        W1 = FREQUENCY * commonCalc;
        /*****************************************
        * This loop should run sequentially
        *****************************************/
        for (j=0; j<ITERS; ++j) {
            
                W2  = h_freqArray[j] * commonCalc;
                DELTA_W = fabs(W1 - W2);
                AVG_W = (W1 + W2) * 0.5;
                val += 2 * cos(DELTA_W * 0.5 * i) * sin(AVG_W * i);
        }
        h_inputFFT[i].x = val / ITERS;
        h_inputFFT[i].y = 0.0;
    }
}

void executeFFT() {

    cufftHandle plan;
    /**********************************************************************************
     * CUDA Libraries expect a device allocated pointer.
     * Make sure that device pointers are visible before calling cuFFT.
     * Check if the pointers exist in the device memory
    **********************************************************************************/
    {
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
    }
    cufftDestroy(plan);
}

void calculateDifference() {
    
    int i;
    /**************************************************************
    * Parallelize the following loop with appropriate data clauses
    ***************************************************************/    
    for (i=0; i<SIGNAL_SIZE; ++i) {
        //Compute the difference
        //Normalize the output of the Inverse FFT by dividing it by SIZE beforehand
        //fabs() is used for finding the absolute value for a double precision number
        //Store the differences in the output array
        h_outputFFT[i].x = fabs(h_inputFFT[i].x - (h_outputFFT[i].x / SIGNAL_SIZE));
    }
}


int main(int argc, char **argv) {

    printf("\n\n==============Execution================\n\n");
    printf("SIGNAL SIZE = %d\n\n", SIGNAL_SIZE);

    //Allocating pointers
    allocate();

    //Initializing
    initialize();
    
    //Executing FFT + IFFT
    executeFFT();
    
    //Calculating difference between inputFFT and inverse outputFFT
    calculateDifference();

    //printMeanError is executed on the CPU but it requires the h_outputFFT array which stores the differences.
    printMeanError(h_outputFFT);
    
    //Deallocating pointers
    deallocate();
    
    printf("\n\n==============Terminated================\n\n");
    return 0;
}

