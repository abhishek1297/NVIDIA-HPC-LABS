program MAIN
    
    use ACCHELPER
    use openacc
    use cufft
    use fileReader
    implicit none

    !For CL Inputs
    include "lib3f.h"
    integer :: i, N, argc
    character*2 :: argv
    
    argc = iargc()
    call getarg(argc, argv)

    N = 1
    if (argc == 1) then
        read (argv, *) N
    end if

    print *, "==============Execution================"
    print *
    print *, "SIGNAL SIZE = ", SIGNAL_SIZE
    print *
    print *, "Times in seconds"

    !Allocate pointers
    allocate(h_inputFFT(SIGNAL_SIZE), h_outputFFT(SIGNAL_SIZE))
    
    !Allocate the device memory before calling any of the functions
    !Copying out the output array which stores the differences that
    !are used by printMeanError funciton

    !$acc data create(h_inputFFT) copyout(h_outputFFT)
    do i=1, N

        !Initializing
        call initialize()
        !Executing FFT + IFFT
        call executeFFT()
        !Calculating difference between inputFFT and inverse outputFFT
        call calculateDifference()
        
        print *, "INIT:", initTime, "FFT:", fftTime, "DIFF:", diffTime
        print *, "TOTAL:", execTime
    end do
    !$acc end data
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
            call loadFrequencies("../../../Common/frequencies.txt", h_freqArray, ITERS)
            
            !pcopyin stands for present or copyin
            !Already loaded data will not be copied
            !parallelize the outer loop
            
            !$acc data pcopyin(h_inputFFT, h_freqArray)
            
            call cpu_time(start)
            
            !$acc parallel loop
            outer:do i=1, SIGNAL_SIZE
                val = 0.0
                commonCalc = TWO_PI * (i-1) / NUM_SAMPLES
                W1 = FREQUENCY * commonCalc
                
                !This loop is executed sequentially by each thread
                !$acc loop seq
                inner:do j=1, ITERS
                    W2  = h_freqArray(j) * commonCalc
                    DELTA_W = DABS(W1 - W2)
                    AVG_W = (W1 + W2) * 0.5
                    val = val + 2 * COS(DELTA_W * 0.5 * (i-1)) * SIN(AVG_W * (i-1))
                end do inner
                h_inputFFT(i) = dcmplx(val / ITERS, 0.0)
            end do outer
            !$acc end parallel
            
            call cpu_time(finish)
            
            !$acc end data

            initTime = finish - start
            execTime = execTime + initTime
        end subroutine initialize

        subroutine executeFFT()
            implicit none

            integer :: plan, err
            
            call cpu_time(start)

            !Create a cufft plan for Z2Z i.e complex to complex in Double precision
            err = cufftPlan1d(plan, SIGNAL_SIZE, CUFFT_Z2Z, 1)
            
            ! CUDA libraries require the GPU allocated data. But the call to cufft is from the host.
            ! host_data construct makes the pointers of GPU allocated data visible to the host.
            ! use_device() clause simply tells the compiler to use the GPU copy instead of CPU copy
            
            !$acc host_data use_device(h_inputFFT, h_outputFFT)

                !Perform FFT out of place transform
                err = err + cufftExecZ2Z(plan, h_inputFFT, h_outputFFT, CUFFT_FORWARD)
                !Perform Inverse FFT in plance transform
                err = err + cufftExecZ2Z(plan, h_outputFFT, h_outputFFT, CUFFT_INVERSE)
            
            !$acc end host_data
            
            err = err + cufftDestroy(plan)

            call cpu_time(finish)

            fftTime = finish - start
            execTime = execTime + fftTime
        end subroutine executeFFT

        subroutine calculateDifference()
            implicit none
            
            integer :: i
            real*8 :: diff
            
            !$acc data pcopyin(h_inputFFT, h_outputFFT)
            
            call cpu_time(start)
            
            !$acc parallel loop
            do i=1, SIGNAL_SIZE
                !normalize the inverse fft output
                diff = DABS(real(h_inputFFT(i) - (real(h_outputFFT(i))/SIGNAL_SIZE)))
                !storing the difference in outputFFT
                h_outputFFT(i) = dcmplx(diff, 0)
            end do
            !$acc end parallel
            
            call cpu_time(finish)

            !$acc end data
            
            diffTime = finish - start
            execTime = execTime + diffTime
        end subroutine calculateDifference
end program MAIN
