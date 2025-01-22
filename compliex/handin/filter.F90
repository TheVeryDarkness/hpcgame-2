
subroutine filter_run(x, wgt, ngrid, is, ie, js, je) bind(c)
    implicit none
    double precision, dimension(0:2699, 0:112), intent(inout) :: x
    double precision, dimension(0:1798, 0:112), intent(in) :: wgt
    integer, dimension(0:113), intent(in) :: ngrid
    integer, value :: is, ie, js, je
    double precision :: tmp(0:ie - is)
    integer :: j, i, p, n, hn

    do j = js, je
        n = ngrid(j)
        hn = (n - 1) / 2
        do i = is, ie
            tmp(i - is) = 0.0d0
            do p = 0, n - 1
                tmp(i - is) = tmp(i - is) + wgt(p, j) * x(i - hn + p, j)
            end do
        end do
        do i = is, ie
            x(i, j) = tmp(i - is)
        end do
    end do
end subroutine filter_run
