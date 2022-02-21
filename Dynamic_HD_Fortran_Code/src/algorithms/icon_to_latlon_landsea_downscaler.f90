module icon_to_latlon_landsea_downscaler

contains

function downscale_coarse_icon_landsea_mask_to_fine_latlon_grid(coarse_landsea_mask, &
                                                                coarse_to_fine_cell_numbers_mapping) &
  result(fine_landsea_mask)
  logical, dimension(:), intent(in) :: coarse_landsea_mask
  integer, dimension(:,:), intent(in) :: coarse_to_fine_cell_numbers_mapping
  logical, dimension(:,:), pointer :: fine_landsea_mask
  integer :: nlat, nlon
  integer :: i,j
  integer :: coarse_cell_number
    nlat = size(coarse_to_fine_cell_numbers_mapping,1)
    nlon = size(coarse_to_fine_cell_numbers_mapping,2)
    allocate(fine_landsea_mask(nlat,nlon))
    do j = 1,nlon
      do i = 1,nlat
        coarse_cell_number = coarse_to_fine_cell_numbers_mapping(i,j)
        fine_landsea_mask(i,j) = coarse_landsea_mask(coarse_cell_number)
      end do
    end do
end function downscale_coarse_icon_landsea_mask_to_fine_latlon_grid

end module icon_to_latlon_landsea_downscaler
