
module ACCHELPER

    implicit none
    
    !Signal generation data
    real, parameter :: TWO_PI  = 2 * 3.14159265358979323846
    integer, parameter :: SIGNAL_SIZE = 67108864
    integer, parameter :: NUM_SAMPLES  = 44100
    integer, parameter :: FREQUENCY  = 100
    integer, parameter :: ITERS  = 100
    
    !store the frequencies loaded from the file
    integer, dimension(ITERS) :: h_freqArray
    
    ! Host/Device pointers used for cufft input and output
    double complex, allocatable, dimension(:) :: h_inputFFT, h_outputFFT
    
    !Timing variables
    real :: initTime, fftTime, diffTime, execTime, start, finish
    
    contains
        subroutine printMeanError(h_errors)
            implicit none

            real*8 :: errorSum = 0.0
            integer :: i
            double complex, allocatable, dimension(:), intent(in) :: h_errors
            
            !Adding all the differences and printing the mean error
            !This loop runs on the CPU
            do i=1, SIGNAL_SIZE
                errorSum = errorSum + real(h_errors(i))
            end do
            print *, "Mean Error: ", errorSum / SIGNAL_SIZE
        end subroutine printMeanError
end module ACCHELPER