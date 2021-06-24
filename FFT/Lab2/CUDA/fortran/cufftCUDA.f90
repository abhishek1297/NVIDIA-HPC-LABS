program MAIN

    use fileReaderF90
    use cudafor
    use cufft
    
    implicit none
    integer, parameter :: SIGNAL_SIZE = 134217728
    double complex, dimension(:), pointer, device :: d_inputFFT, d_outputFFT, d_outputInvFFT
    double complex, dimension(:), pointer :: h_inputFFT, h_outputInvFFT
    
    type(cudaEvent) :: start, stop
    real :: execTime
    integer :: err

    !Creating events
    err = cudaEventCreate(start)
    err = cudaEventCreate(stop)

    allocate(h_inputFFT(SIGNAL_SIZE), h_outputInvFFT(SIGNAL_SIZE),&
    d_inputFFT(SIGNAL_SIZE), d_outputFFT(SIGNAL_SIZE), d_outputInvFFT(SIGNAL_SIZE))
    
    call initialize()
    
    call executeFFT()
    
    print *,"Only FFT:"
    print *, "Execution Time = ", execTime

    deallocate(h_inputFFT, h_outputInvFFT, &
    d_inputFFT, d_outputFFT, d_outputInvFFT)

    err = cudaEventDestroy(start)
    err = cudaEventDestroy(stop)

    contains

        subroutine initialize()
            implicit none
            
            integer :: i
            real*8, dimension(SIGNAL_SIZE) :: inputs
            
            ! Loading real numbers from a fle
            call readInputs("../../Inputs/complexInputs.txt", inputs, SIGNAL_SIZE);
            
            do i=1, SIGNAL_SIZE
                h_inputFFT(i) = dcmplx(inputs(i), 0.0)
            end do
            
            !cudaMemcpy
            d_inputFFT = h_inputFFT
        end subroutine initialize

        subroutine executeFFT()
            implicit none
            
            integer :: i, plan, err
            real*8 :: errorSum
            errorSum = 0.0
            
            err = cudaEventRecord(start, 0)
            !Create a cufft plan for Z2Z i.e complex to complex in Double precision
            err = cufftPlan1d(plan, SIGNAL_SIZE, CUFFT_Z2Z, 1)
            
            !Perform FFT out of place transform
            err = err + cufftExecZ2Z(plan, d_inputFFT, d_outputFFT, CUFFT_FORWARD)
            
            !Perform Inverse FFT in plance transform
            err = err + cufftExecZ2Z(plan, d_outputFFT, d_outputInvFFT, CUFFT_INVERSE)
            
            err = err + cufftDestroy(plan)
            
            !Cuda memcpy
            h_outputInvFFT = d_outputInvFFT
            
            err = cudaEventRecord(stop, 0)
            err = cudaEventSynchronize(stop)

            !Accumulate total error
            do i=1, SIGNAL_SIZE
                errorSum  = errorSum + DABS(real(h_inputFFT(i)) - real(h_outputInvFFT(i))/SIGNAL_SIZE)
            end do

            print *, "Mean Error: ", errorSum / SIGNAL_SIZE
            print *
            
            err = cudaEventElapsedTime(execTime, start, stop)
            execTime = execTime / 1000
        end subroutine executeFFT
    
end program MAIN
