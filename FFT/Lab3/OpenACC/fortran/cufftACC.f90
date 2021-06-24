program MAIN
    
    !Contains all the required pointers, variables, and function
    use ACCHELPER
    
    use openacc
    use cufft
    use fileReader
    
    implicit none
    
    print *, "==============Execution================"
    print *
    print *, "SIGNAL SIZE = ", SIGNAL_SIZE
    print *

    !Allocate pointers
    allocate(h_inputFFT(SIGNAL_SIZE), h_outputFFT(SIGNAL_SIZE))
    
    !Initializing
    call initialize()
    
    !Executing FFT + IFFT
    call executeFFT()
    
    !Calculating difference between inputFFT and inverse outputFFT
    call calculateDifference()
    
    print *
    
    !printMeanError is executed on the CPU but it requires the h_outputFFT array which stores the differences
    call printMeanError(h_outputFFT)

    !Deallocating pointers
    deallocate(h_inputFFT, h_outputFFT)
    
    print *
    print *, "==============Terminated================"
    print *

    contains

        subroutine initialize()

            !Initializing the complex number
            !with a random wave using Openacc directives
            
            implicit none
            integer :: i, j
            real*8 :: val, commonCalc, W1, W2, DELTA_W, AVG_W
           
            !reading a file and loading the frequencies
            call loadFrequencies("../../Common/frequencies.txt", h_freqArray, ITERS)
            
            !**************************************************************
            !Parallelize the following loop with appropriate data clauses
            !**************************************************************
            
            outer:do i=1, SIGNAL_SIZE
                
                val = 0.0
                commonCalc = TWO_PI * (i-1) / NUM_SAMPLES
                W1 = FREQUENCY * commonCalc
                
                !****************************************
                !This loop should run sequentially
                !****************************************
                inner:do j=1, ITERS
                    W2  = h_freqArray(j) * commonCalc
                    DELTA_W = DABS(W1 - W2)
                    AVG_W = (W1 + W2) * 0.5
                    val = val + 2 * COS(DELTA_W * 0.5 * (i-1)) * SIN(AVG_W * (i-1))
                end do inner
                h_inputFFT(i) = dcmplx(val / ITERS, 0.0)
            end do outer
            
        end subroutine initialize

        subroutine executeFFT()
            implicit none

            integer :: plan, err
            !**********************************************************************************
            !CUDA Libraries expect a device allocated pointer.
            !Make sure that device pointers are visible before calling cuFFT.
            !Check if the pointers exist in the device memory
            !**********************************************************************************/
            
            !***********************************************************************
            !Do the following steps
               
            !   1. Create a 1D plan of length SIGNAL_SIZE in double precision
            !   for Complex to Complex Transform.
       
            !   2. Execute FFT (time to frequency) for given input signal
            !       -Do an out of place transform
            !       i.e use two different arrays for input and output
               
            !   3. Execute Inverse FFT (frequency to time) for the given output of fft
            !       -Do an in-place transform
            !       i.e use the same array for input and output. Use output array
            !       -Input array must not be updated
               
            !***********************************************************************
            
            err = err + cufftDestroy(plan)
        
        end subroutine executeFFT

        subroutine calculateDifference()
            implicit none
            
            integer :: i
            real*8 :: diff
            
            !**************************************************************
            !Parallelize the following loop with appropriate data clauses
            !**************************************************************
            do i=1, SIGNAL_SIZE
                !normalize the inverse fft output
                diff = DABS(real(h_inputFFT(i) - (real(h_outputFFT(i))/SIGNAL_SIZE)))
                !storing the difference in outputFFT
                h_outputFFT(i) = dcmplx(diff, 0)
            end do
        end subroutine calculateDifference
end program MAIN
