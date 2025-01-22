
subroutine filter_run(x, wgt, ngrid, is, ie, js, je) bind(c)
    implicit none
    double precision, dimension(2700, 113), intent(inout) :: x
    double precision, dimension(1799, 113), intent(in) :: wgt
    integer, dimension(113), intent(in) :: ngrid
    integer, value :: is, ie, js, je
    double precision :: tmp(ie - is + 1)
    integer :: j, i, p, n, hn

    do j = js, je
        n = ngrid(j)
        hn = (n - 1) / 2
        do i = is, ie
            tmp(i - is + 1) = 0.0d0
            do p = 0, n - 1
                tmp(i - is + 1) = tmp(i - is + 1) + wgt(p + 1, j) * x(i - hn + p + 1, j)
            end do
        end do
        do i = is, ie
            x(j, i) = tmp(i - is + 1)
        end do
    end do
end subroutine filter_run
