module fileReader

    implicit none
    contains
        subroutine loadFrequencies(fname, h_freqArray, N)
            implicit none
            integer :: i
            character*(*), intent(in) :: fname
            integer, intent(in) :: N
            integer, dimension(:), intent(out) :: h_freqArray
            open(unit=9, file=trim(fname))
            do i=1, N
                read (9,*) h_freqArray(i)
            end do
            close(9)
        end subroutine loadFrequencies

end module fileReader