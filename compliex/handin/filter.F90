subroutine filter_run(x, wgt, ngrid, is, ie, js, je) bind(c)
    implicit none
    double precision, dimension(113, 2700) :: x
    double precision, dimension(113, 1799) :: wgt
    integer, dimension(113) :: ngrid
    integer :: is, ie, js, je
    double precision :: tmp(ie - is + 1)
    integer :: j, i, p, n, hn

    do j = js, je
        n = ngrid(j)
        hn = (n - 1) / 2
        do i = is, ie
            tmp(i - is + 1) = 0.0d0
            do p = 0, n - 1
                tmp(i - is + 1) = tmp(i - is + 1) + wgt(j, p + 1) * x(j, i - hn + p + 1)
            end do
        end do
        do i = is, ie
            x(j, i) = tmp(i - is + 1)
        end do
    end do
end subroutine filter_run
