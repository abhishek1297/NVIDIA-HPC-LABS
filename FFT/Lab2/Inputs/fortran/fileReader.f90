module fileReaderF90

    implicit none
    contains
        subroutine readInputs(fname, inputs, N)
            implicit none
            integer :: i
            character*(*), intent(in) :: fname
            integer, intent(in) :: N
            real*8 , dimension(:), intent(out) :: inputs
            open(unit=9, file=trim(fname))
            do i=1, N
                read (9,*) inputs(i)
            end do
            close(9)
        end subroutine readInputs

end module fileReaderF90