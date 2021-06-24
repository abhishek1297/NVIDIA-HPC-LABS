module CUHELPER

    use cudafor
    implicit none

    !Signal generation data
    real, parameter :: TWO_PI  = 2 * 3.14159265358979323846
    integer, parameter :: SIGNAL_SIZE = 67108864
    integer, parameter :: NUM_SAMPLES  = 44100
    integer, parameter :: FREQUENCY  = 100
    integer, parameter :: ITERS  = 100

    !Device allocated pointers
    double complex, allocatable, dimension(:), device :: d_inputFFT, d_outputFFT
    integer, allocatable, dimension(:), device :: d_freqArray

    !Host allocated pointers
    double complex, allocatable, dimension(:) :: h_errors
    
    !Timing variables
    type(cudaEvent) :: start, stop
    real :: execTime, initTime, fftTime, diffTime

    contains
        subroutine printMeanError(h_errors)
        
            implicit none
            
            double complex, allocatable, dimension(:), intent(in) :: h_errors
            real*8 :: errorSum = 0.0
            integer :: i
            
            !Adding all the differences and printing the mean error
            !This loop runs on the CPU
            do i=1, SIGNAL_SIZE
                errorSum = errorSum + real(h_errors(i))
            end do
            
            print *, "Mean Error: ", errorSum / SIGNAL_SIZE
        end subroutine printMeanError    
end module CUHELPER