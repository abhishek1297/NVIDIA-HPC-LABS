#include "../../../Common/c/fileReader.h"
#include "../../../Common/c/acchelper.h"
#include <unistd.h>
//CUDA includes
#include <cufft.h>

//Size of array in bytes
extern size_t MEM_SIZE;

//Host allocated pointers
extern Complex *h_inputFFT, *h_outputFFT;

//Timing variables
extern struct timeval start, stop;
float execTime;


void initialize() {

    /**
     * Initializing the complex number
     * with a random wave using Openacc directives
     */
    float time;
    int i, j;
    double val, commonCalc, W1, W2, DELTA_W, AVG_W;
    int h_freqArray[ITERS];

    //Loading the frequencies from a file.
    loadFrequencies("../../../Common/frequencies.txt", h_freqArray, ITERS);
    
    // pcopyin is short for present_or_copyin
    // Whatever that is not on the device will be copied.
    #pragma acc data pcopyin(h_inputFFT[:SIGNAL_SIZE], h_freqArray[:ITERS])
    {
        gettimeofday(&start, NULL);
        
        #pragma acc parallel loop
        for (i=0; i<SIGNAL_SIZE; ++i) {
            val = 0.0;
            commonCalc = TWO_PI * i / NUM_SAMPLES;
            W1 = FREQUENCY * commonCalc;
            
            #pragma acc loop seq
            for (j=0; j<ITERS; ++j) {
                
                    W2  = h_freqArray[j] * commonCalc;
                    DELTA_W = fabs(W1 - W2);
                    AVG_W = (W1 + W2) * 0.5;
                    val += 2 * cos(DELTA_W * 0.5 * i) * sin(AVG_W * i);
            }
            h_inputFFT[i].x = val / ITERS;
            h_inputFFT[i].y = 0.0;
        }

        gettimeofday(&stop, NULL);
    }

    time = (float)(stop.tv_usec - start.tv_usec) / 1000000 + (float)(stop.tv_sec - start.tv_sec);
    execTime += time;
    printf("INIT:%.2f\t", time);
}

void executeFFT() {

    float time;
    //Creating a plan for Complex to Complex operation in double precision.
    cufftHandle plan;

    gettimeofday(&start, NULL);
    
    cufftPlan1d(&plan, SIGNAL_SIZE, CUFFT_Z2Z, 1); // Z2Z is C2C with Double Precision
    /**
     * CUDA libraries require the GPU allocated data. But the call to cufft is from the host.
     * host_data construct makes the pointers of GPU allocated data visible to the host.
     * use_device() clause simply tells the compiler to use the GPU copy instead of CPU copy
     */
    #pragma acc host_data use_device(h_inputFFT, h_outputFFT)
    {
        //Perform FFT
        cufftExecZ2Z(plan, (cufftDoubleComplex *)h_inputFFT,
                        (cufftDoubleComplex *)h_outputFFT,
                        CUFFT_FORWARD);

        //Perform Inverse FFT
        cufftExecZ2Z(plan, (cufftDoubleComplex *)h_outputFFT,
                        (cufftDoubleComplex *)h_outputFFT,
                        CUFFT_INVERSE);
    }

    cufftDestroy(plan);
    
    gettimeofday(&stop, NULL);
    
    time = (float)(stop.tv_usec - start.tv_usec) / 1000000 + (float)(stop.tv_sec - start.tv_sec);
    execTime += time;
    printf("FFT:%.2f\t", time);
}

void calculateDifference() {
    
    float time;
    int i;
    //Data is already present on the GPU
    // Using the output array to store the differences
    #pragma acc data pcopyin(h_inputFFT[:SIGNAL_SIZE], h_outputFFT[:SIGNAL_SIZE])
    {
        gettimeofday(&start, NULL);
        
        #pragma acc parallel loop
        for (i=0; i<SIGNAL_SIZE; ++i) {
            //Compute the difference
            //Normalize the output of the Inverse FFT by dividing it by SIZE beforehand
            //Store the differences in the output array
            h_outputFFT[i].x = fabs(h_inputFFT[i].x - (h_outputFFT[i].x / SIGNAL_SIZE));
        }
        
        gettimeofday(&stop, NULL);
    }
    
    time = (float)(stop.tv_usec - start.tv_usec) / 1000000 + (float)(stop.tv_sec - start.tv_sec);
    execTime += time;
    printf("DIFF:%.2f\t", time);
}


int main(int argc, char **argv) {

    int i, N = 1;
    // Number of times to execute the parallel portions of the code
    // To run ./fftacc.out <N> or make ARGS="<N>"
    //By default the N = 1
    if (argc > 1) { 
        N = atoi(argv[1]);
    }
    printf("\n\n==============Execution================\n\n");
    printf("SIGNAL SIZE = %d\n\n", SIGNAL_SIZE);

    //Allocating pointers
    allocate();

    printf("Times in seconds\n\n");

    //Allocating device memory before doing any operations.
    #pragma acc data create(h_inputFFT[:SIGNAL_SIZE]) copyout(h_outputFFT[:SIGNAL_SIZE])
    {
        for (i=0; i<N; ++i) {
        
            //Initializing
            initialize();
            //Executing FFT + IFFT
            executeFFT();
            //Calculating difference between inputFFT and inverse outputFFT
            calculateDifference();
            printf("TOTAL:%.2f\n\n", execTime);
		sleep(10);
        }
    }

    //printMeanError is executed on the CPU but it requires the h_outputFFT array which stores the differences.
    printMeanError(h_outputFFT);
    
    //Deallocating pointers
    deallocate();

    printf("\n\n==============Terminated================\n\n");
    return 0;
}

