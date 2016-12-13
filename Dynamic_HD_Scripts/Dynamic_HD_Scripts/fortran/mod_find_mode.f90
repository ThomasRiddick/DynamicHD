module mod_find_mode

!Module contains function that reduces the first dimension of an array by finding the mode of a
!slice of the first dimension (and is a single point in the remaining dimensions) and using that
!as the new value for the position of that slice. Contains a subroutine for reducing a four
!dimensional array and a subroutine for reducing a three dimensional array

implicit none

contains

subroutine find_mode_4d(array_in,array_out,naxis1,naxis2,naxis3,naxis4)

!Reduces the first dimension of a four dimensional array to the mode of that
!dimension for slices spanning that dimension at each point in the remaining dimensions

implicit none

!Variables with intent IN
!Dimensions of the input array
integer, intent(in) :: naxis1, naxis2, naxis3, naxis4
!Input array
integer, intent(in), dimension(naxis1,naxis2,naxis3,naxis4) :: array_in

!Variables with intent OUT
!Output array
integer, intent(out), dimension(naxis2,naxis3,naxis4) :: array_out

!Local variables
!List of counts of unique values found in a slice
integer, allocatable, dimension(:) :: unique_nums
!highest number in the input array
integer :: max_num
!counters
integer :: i,j,k,l
    !Assuming all values are non-negative this (+1) is the maximum number of elements
    !the list of unique values requires
    max_num = MAXVAL(array_in)
    !Create a static list of counts of unique values and initialize to zero
    allocate(unique_nums(max_num+1))
    unique_nums = 0
    !Iterate over each point in all the dimensions other than the first
    do l = 1,naxis4
        do k = 1,naxis3
            do j = 1,naxis2
                !Iterate over a slice of the first dimension
                do i = 1,naxis1
                    !Ignore negative values
                    if (array_in(i,j,k,l) < 0) cycle
                    !Increment count for this value of this element, add offset of 1 as count
                    !of the value zero is held in position 1
                    unique_nums(array_in(i,j,k,l) + 1) = unique_nums(array_in(i,j,k,l) + 1) + 1
                end do
                !Allocate the value of this point in the output array as the highest count found
                !in the list of unique values and then reset the array to 0 for the next cycle
                array_out(j,k,l) = MAXLOC(unique_nums,1) - 1
                unique_nums = 0
            end do
        end do
    end do
    deallocate(unique_nums)
end subroutine find_mode_4d

subroutine find_mode_3d(array_in,array_out,naxis1,naxis2,naxis3)

!Reduces the first dimension of a three dimensional array to the mode of that
!dimension for slices spanning that dimension at each point in the remaining dimensions

implicit none

!Variables with intent IN
!Dimensions of the input array
integer, intent(in) :: naxis1, naxis2, naxis3
!Input array
integer, intent(in), dimension(naxis1,naxis2,naxis3) :: array_in
!Variables with intent OUT
!Output array
integer, intent(out), dimension(naxis2,naxis3) :: array_out

!Local variables
!List of counts of unique values found in a slice
integer, allocatable, dimension(:) :: unique_nums
!highest number in the input array
integer :: max_num
!counters
integer :: i,j,k
    !Assuming all values are non-negative this (+1) is the maximum number of elements
    !the list of unique values requires
    max_num = MAXVAL(array_in)
    !Create a static list of counts of unique values and initialize to zero
    allocate(unique_nums(max_num+1))
    unique_nums = 0
    !Iterate over each point in all the dimensions other than the first
    do k = 1,naxis3
        do j = 1,naxis2
            !Iterate over a slice of the first dimension
            do i = 1,naxis1
                !Ignore negative values
                if (array_in(i,j,k) < 0) cycle
                !Increment count for this value of this element, add offset of 1 as count
                !of the value zero is held in position 1
                unique_nums(array_in(i,j,k)+1) = unique_nums(array_in(i,j,k)+1) + 1
            end do
            array_out(j,k) = MAXLOC(unique_nums,1) - 1
            unique_nums = 0
        end do
    end do
    deallocate(unique_nums)
end subroutine find_mode_3d

end module mod_find_mode
