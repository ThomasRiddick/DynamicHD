module mod_check_for_value

!Module contains function that reduce the first dimension of an array by search for a value in slices
!of the array spanning this dimension (one for each point in the space formed by the remaining set of
!dimensions) and reducing slice to that value if found or zero if it is not found. There are two
!versions the function that performs this reduction contained, one for a 4 dimensional array another
!for a three dimensional array.

implicit none

contains

subroutine check_for_value_4d(array_in,array_out,value,naxis1,naxis2,naxis3,naxis4)

!Reduces the first dimension of a 4d array to value if that value is present in a slice of the array
!spanning the last dimension, otherwise to 0

implicit none

!Variables with intent IN
!Dimensions of the input array
integer, intent(in) :: naxis1, naxis2, naxis3, naxis4
!Input array
integer, intent(in), dimension(naxis1,naxis2,naxis3,naxis4) :: array_in
!Value to check for in the first dimension and reduce to if found
integer, intent(in) :: value

!Variables with intent OUT
!Output array
integer, intent(out), dimension(naxis2,naxis3,naxis4) :: array_out

!Local variables
!A point in the slice across the first dimension that has the value
!being search for has been found
logical :: value_found
!Loop counters
integer :: i,j,k,l
    value_found = .FALSE.
    !Loop over other dimensions
    do l = 1,naxis4
        do k = 1,naxis3
            do j = 1,naxis2
                !Loop over first dimension
                do i = 1,naxis1
                    if (array_in(i,j,k,l) == value) value_found = .TRUE.
                end do
                !If a value was found set it in the reduce array appropriately or
                !otherwise set zero
                if (value_found) then
                    array_out(j,k,l) = value
                    value_found = .FALSE.
                else
                    array_out(j,k,l) = 0
                end if
            end do
        end do
    end do
end subroutine check_for_value_4d

subroutine check_for_value_3d(array_in,array_out,value,naxis1,naxis2,naxis3)

!Reduces the last dimension of a 3d array to value if that value is present in a slice of the array
!spanning the last dimension, otherwise to 0

implicit none

!Variables with intent IN
integer, intent(in) :: naxis1, naxis2, naxis3
!Input array
integer, intent(in), dimension(naxis1,naxis2,naxis3) :: array_in
!Value to check for in the first dimension and reduce to if found
integer, intent(in) :: value

!Variables with intent OUT
!Output array
integer, intent(out), dimension(naxis2,naxis3) :: array_out

!Local variables
!A point in the slice across the first dimension that has the value
!being search for has been found
logical :: value_found
!Loop counters
integer :: i,j,k
    value_found = .FALSE.
    !Loop over other dimensions
    do k = 1,naxis3
        do j = 1,naxis2
            !Loop over first dimension
            do i = 1,naxis1
                if (array_in(i,j,k) == value) value_found = .TRUE.
            end do
            !If a value was found set it in the reduce array appropriately or
            !otherwise set zero
            if (value_found) then
                array_out(j,k) = value
                value_found = .FALSE.
            else
                array_out(j,k) = 0
            end if
        end do
    end do
end subroutine check_for_value_3d

end module mod_check_for_value
