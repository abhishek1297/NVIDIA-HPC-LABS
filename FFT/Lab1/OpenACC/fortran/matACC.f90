program MAIN

    use openacc
    implicit none
    

    integer, parameter :: SIZE_ = 16384
    
    !Host/Device pointers
    real*8, allocatable, dimension(:,:) :: mat, matT, matSym

    !CLI argument header file
    include "lib3f.h"
    character*2 :: argv
    integer :: i, N, TILE_DIM, argc
    real*8 ::execTime, start, finish
    
    N = 1
    TILE_DIM = 32
    execTime = 0

    argc = iargc()
    call getarg(argc, argv)

    if (argc == 1) then
        read (argv, *) N
    end if

    print *, SIZE_, "x", SIZE_, "Matrix"
    print *
    
    allocate(mat(SIZE_,SIZE_), matT(SIZE_,SIZE_), matSym(SIZE_,SIZE_))
    

    print *,"Execution times(sec)"

    !copying out to print if needed.
    !$acc data copyout(mat, matT, matSym)
    do i=1,N

        call cpu_time(start)

        call initialize()
        
        call calculateSymmetricMatrix()
        
        call cpu_time(finish)

        execTime = execTime + (finish - start)
        print *,"At", i, execTime
    end do
    !$acc end data
    
    deallocate(mat, matT, matSym)


    contains
        subroutine printMatrix(mat)
            implicit none
            real*8, allocatable, dimension(:,:), intent(in) :: mat
            integer :: i, j
            if (SIZE_ > 8) then
                return
            end if
            do j=1, SIZE_
                do i=1, SIZE_
                    print *, mat(i,j)
                end do
            end do
        end subroutine printMatrix
        subroutine initialize()
            implicit none
            integer :: i, j
            
            !Tile provides better data locality
            !$acc parallel loop tile(32, 32) present(mat)
            do j=1, SIZE_
                do i=1, SIZE_
                    mat(i, j) = (j-1) * SIZE_ + (i-1)
                end do
            end do
            !acc end parallel

        end subroutine initialize
        subroutine transpose()
            implicit none
            integer :: i, j
            
            !$acc parallel loop tile(32, 32) present(mat, matT)
            do j=1, SIZE_
                do i=1, SIZE_
                    matT(j, i) = mat(i, j)
                end do
            end do
            !acc end parallel
        end subroutine transpose
        subroutine matrixMultiply()
            implicit none
            integer :: i, j, k
            real*8 :: accum
            
            !$acc parallel loop tile(32, 32) present(mat, matT, matSym)
            do j=1, SIZE_
                do i=1, SIZE_
                    accum = 0
                    do k=1, SIZE_
                        accum = accum + (mat(i, k) * matT(k, j))
                    end do
                    matSym(i, j) = accum
                end do
            end do
            !acc end parallel
        end subroutine matrixMultiply
        subroutine calculateSymmetricMatrix()
            implicit none
            call transpose()
            call matrixMultiply()
        end subroutine calculateSymmetricMatrix
end program MAIN
