module landsea_downscaler_driver_mod

use icon_to_latlon_landsea_downscaler
use precision_mod
implicit none

contains

  subroutine landsea_downscaler_icosohedral_cell_latlon_pixel_f2py_wrapper( &
      coarse_landsea_mask,coarse_to_fine_cell_numbers_mapping, &
      fine_landsea_mask,nlat_fine,nlon_fine,ncells_coarse)
    integer, intent(in) :: nlat_fine,nlon_fine
    integer, intent(in) :: ncells_coarse
    integer, intent(in), dimension(ncells_coarse) :: coarse_landsea_mask
    integer, intent(in), dimension(nlat_fine,nlon_fine) :: coarse_to_fine_cell_numbers_mapping
    logical, dimension(:,:), pointer :: fine_landsea_mask_logical
    integer, intent(out), dimension(nlat_fine,nlon_fine) :: fine_landsea_mask
      fine_landsea_mask_logical => &
        downscale_coarse_icon_landsea_mask_to_fine_latlon_grid((coarse_landsea_mask == 1), &
                                                               coarse_to_fine_cell_numbers_mapping)
      where (fine_landsea_mask_logical)
        fine_landsea_mask = 1
      else where
        fine_landsea_mask = 0
      end where
      deallocate(fine_landsea_mask_logical)
  end subroutine landsea_downscaler_icosohedral_cell_latlon_pixel_f2py_wrapper

end module landsea_downscaler_driver_mod
