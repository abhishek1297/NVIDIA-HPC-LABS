module FFTW3
    use, intrinsic :: iso_c_binding
    include 'fftw3.f'
end module FFTW3


program MAIN

    use omp_lib
    use FFTW3
    use fileReaderF90

    implicit none
    integer, parameter :: SIZE_ = 134217728
    real*8 :: execTime, start, finish

    ! I/O pointers
    double complex, dimension(:), pointer :: inputFFT, outputFFT
    integer*8 plan
    integer :: argc
    character*2 :: argv
    
    ! Use -m flag to execute multicore version
    ! ./ftn_fftw.out -m or make fortran ARGS="-m"
    argc = iargc()
    call getarg(argc, argv)

    allocate(inputFFT(SIZE_), outputFFT(SIZE_))
    
    call initialize()

    call cpu_time(start)

    if (argc == 1 .and. argv == "-m") then
        print *, "Multicore Execution"
        call multicoreFFT()
    else
        print *, "Unicore Execution"
        call executeFFT()
    end if

    call cpu_time(finish)
    
    execTime = finish - start

    print *
    print *, "Execution Time(sec) = ", execTime
    print *
    deallocate(inputFFT, outputFFT)

    contains
        subroutine initialize()
            implicit none
            integer :: i
            real*8, dimension(SIZE_) :: inputs
            
            print *,"Loading File..."    
            !Loading real numbers from a fle
            call readInputs("../../Inputs/complexInputs.txt", inputs, SIZE_);
            
            do i=1, SIZE_
                inputFFT(i) = dcmplx(inputs(i), 0.0)
            end do
        end subroutine initialize

        subroutine executeFFT()
            implicit none
            
            integer :: i
            real*8 :: error
            error = 0.0

            ! Creating a plan for FFT FORWARD
            call dfftw_plan_dft_1d(plan, SIZE_, inputFFT, outputFFT, FFTW_FORWARD, FFTW_ESTIMATE)
            
            ! Performing FFT FORWARD
            call dfftw_execute_dft(plan, inputFFT, outputFFT)
            
            ! Creating a plan for FFT INVERSE
            call dfftw_plan_dft_1d(plan, SIZE_, outputFFT, outputFFT, FFTW_BACKWARD, FFTW_ESTIMATE)
            
            ! Performing FFT INVERSE
            call dfftw_execute_dft(plan, outputFFT, outputFFT)
            call dfftw_destroy_plan(plan)
    
            ! Accumulating total error
            do i=1, SIZE_
                error = error + DABS(real(inputFFT(i)) - real(outputFFT(i)) / SIZE_)
            end do
            print *
            print *, "Mean Error = ", error / SIZE_
        end subroutine executeFFT

        subroutine multicoreFFT()
            implicit none
            
            integer :: err, threads
            threads = omp_get_max_threads()
            print *, "***Total CPU THREADS =", threads
            
            call dfftw_init_threads(err)
            
            if (err == 0) then
                print *, "Error: Initializing Threads failed"
                call exit(1)
            end if
            
            call dfftw_plan_with_nthreads(threads)
            
            !Now the following FFT call will be planned with multi-threading
            call executeFFT()
            
            call dfftw_cleanup_threads()

        end subroutine multicoreFFT
end program MAIN 
