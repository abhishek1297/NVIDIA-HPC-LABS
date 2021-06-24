program MAIN
    
    use fileReaderF90
    use cufft

    implicit none
    integer, parameter :: SIGNAL_SIZE = 134217728
    double complex, allocatable, dimension(:) :: h_inputFFT, h_outputFFT, h_outputInvFFT
    real*8 :: execTime, start, finish
    
    allocate(h_inputFFT(SIGNAL_SIZE), h_outputFFT(SIGNAL_SIZE), h_outputInvFFT(SIGNAL_SIZE))
    
    call initialize() 
    
    !$acc data copyin(h_inputFFT) create(h_outputFFT, h_outputInvFFT)
    call executeFFT()
    !$acc end data
    
    print *, "Only FFT:"
    print *, "Execution Time:", execTime
    print *

    deallocate(h_inputFFT, h_outputFFT, h_outputInvFFT)





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
        end subroutine initialize

        subroutine executeFFT()
            implicit none
            
            integer :: i, plan, err
            real*8 :: errorSum
            errorSum = 0.0
            
            call cpu_time(start)

            !Create a cufft plan for Z2Z i.e complex to complex in Double precision
            err = cufftPlan1d(plan, SIGNAL_SIZE, CUFFT_Z2Z, 1)

            !$acc host_data use_device(h_inputFFT, h_outputFFT, h_outputInvFFT)
            
            
            !Perform FFT out of place transform
            err = err + cufftExecZ2Z(plan, h_inputFFT, h_outputFFT, CUFFT_FORWARD)
            
            !Perform Inverse FFT in plance transform
            err = err + cufftExecZ2Z(plan, h_outputFFT, h_outputInvFFT, CUFFT_INVERSE)
            !$acc end host_data

            err = err + cufftDestroy(plan)

            !copy to host
            !$acc update self(h_outputInvFFT(:SIGNAL_SIZE)) wait
            
            call cpu_time(finish)
    
            !Accumulate total error
            do i=1, SIGNAL_SIZE
                errorSum  = errorSum + DABS(real(h_inputFFT(i)) - real(h_outputInvFFT(i))/SIGNAL_SIZE)
            end do
            
            print *
            print *, "Mean Error: ", errorSum / SIGNAL_SIZE
            print *

            execTime = finish - start
        end subroutine executeFFT
    
end program MAIN
