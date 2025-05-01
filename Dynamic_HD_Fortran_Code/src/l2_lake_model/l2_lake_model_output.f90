module l2_lake_model_output

use netcdf
use check_return_code_netcdf_mod
use parameters_mod
use l2_calculate_lake_fractions_mod, only: dp

implicit none

contains

subroutine write_lake_volumes_field(lake_volumes_filename,&
                                    lake_volumes,_NPOINTS_LAKE_)
  character(len = *), intent(in) :: lake_volumes_filename
  real(dp), pointer, dimension(_DIMS_), intent(in) :: lake_volumes
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
  _DEF_NPOINTS_LAKE_ _INTENT_in_
    call check_return_code(nf90_create(lake_volumes_filename,nf90_noclobber,ncid))
    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_def_dim(ncid, "lat",nlat_lake,lat_dimid))
    call check_return_code(nf90_def_dim(ncid, "lon",nlon_lake,lon_dimid))
    dimids = (/lon_dimid,lat_dimid/)
    _END_IF_USE_SINGLE_INDEX_
    call check_return_code(nf90_def_var(ncid, "lake_volumes_field", nf90_real, dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    _IF_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,lake_volumes))
    _ELSE_IF_NOT_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,transpose(lake_volumes)))
    _END_IF_USE_LONLAT_
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_volumes_field

subroutine write_lake_numbers_field(working_directory,&
                                    lake_numbers,timestep,&
                                    _NPOINTS_LAKE_)
  character(len = *), intent(in) :: working_directory
  integer, dimension(_DIMS_), pointer, intent(in) :: lake_numbers
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
  _DEF_NPOINTS_LAKE_ _INTENT_in_
    if(timestep == -1) then
      filename = trim(working_directory) // 'lake_model_results.nc'
    else if (timestep == -2) then !Use working directory variable to hold a filename
      filename = trim(working_directory)
    else
      filename = trim(working_directory) // 'lake_model_results_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_def_dim(ncid,"lat",nlat_lake,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",nlon_lake,lon_dimid))
    dimids = (/lon_dimid,lat_dimid/)
    _END_IF_USE_SINGLE_INDEX_
    call check_return_code(nf90_def_var(ncid,"lake_number",nf90_real,dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    _IF_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,&
                                        lake_numbers))
    _ELSE_IF_NOT_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,&
                                        transpose(lake_numbers)))
    _END_IF_USE_LONLAT_
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_numbers_field

subroutine write_diagnostic_lake_volumes_field(working_directory, &
                                               diagnostic_lake_volumes, &
                                               timestep,_NPOINTS_LAKE_)
  character(len = *), intent(in) :: working_directory
  real(dp), dimension(_DIMS_), pointer, intent(in) :: diagnostic_lake_volumes
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
  _DEF_NPOINTS_LAKE_ _INTENT_in_
    if(timestep == -1) then
      filename = trim(working_directory) // '/diagnostic_lake_volume_results.nc'
    else
      filename = trim(working_directory) // '/diagnostic_lake_volume_results_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_def_dim(ncid,"lat",nlat_lake,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",nlon_lake,lon_dimid))
    dimids = (/lon_dimid,lat_dimid/)
    _END_IF_USE_SINGLE_INDEX_
    call check_return_code(nf90_def_var(ncid,"diagnostic_lake_volumes",nf90_real,dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    _IF_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,diagnostic_lake_volumes))
    _ELSE_IF_NOT_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,transpose(diagnostic_lake_volumes)))
    _END_IF_USE_LONLAT_
    call check_return_code(nf90_close(ncid))
end subroutine write_diagnostic_lake_volumes_field

subroutine write_lake_fractions_field(working_directory, &
                                      lake_fraction_on_surface_grid, &
                                      timestep, &
                                      _NPOINTS_SURFACE_)
  character(len = *), intent(in) :: working_directory
  real(dp), dimension(_DIMS_), allocatable, intent(in) :: lake_fraction_on_surface_grid
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
    if(timestep == -1) then
      filename = trim(working_directory) // '/lake_fractions.nc'
    else if (timestep == -2) then !Use working directory variable to hold a filename
      filename = trim(working_directory)
    else
      filename = trim(working_directory) // '/lake_fractions_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_def_dim(ncid,"lat",nlat_surface,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",nlon_surface,lon_dimid))
    dimids = (/lon_dimid,lat_dimid/)
    _END_IF_USE_SINGLE_INDEX_
    call check_return_code(nf90_def_var(ncid,"lake_fraction",nf90_real,dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    _IF_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,&
                                        lake_fraction_on_surface_grid))
    _ELSE_IF_NOT_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,varid,&
                                        transpose(lake_fraction_on_surface_grid)))
    _END_IF_USE_LONLAT_
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_fractions_field

subroutine write_binary_lake_mask_field(binary_lake_mask_filename, &
                                        number_fine_grid_cells, &
                                        lake_pixel_counts_field, &
                                        lake_fractions_field, &
                                        binary_lake_mask, &
                                        _NPOINTS_SURFACE_)
  character(len = *), intent(in) :: binary_lake_mask_filename
  integer, dimension(_DIMS_), pointer, intent(in)  :: number_fine_grid_cells
  integer, dimension(_DIMS_), pointer, intent(in)  :: lake_pixel_counts_field
  real(dp), dimension(_DIMS_), pointer, intent(in) :: lake_fractions_field
  logical, dimension(_DIMS_), pointer, intent(in)  :: binary_lake_mask
  integer, dimension(_DIMS_), pointer :: binary_lake_mask_int
  integer :: ncid,lat_dimid,lon_dimid
  integer :: number_fine_grid_cells_varid,lake_pixel_counts_varid
  integer :: lake_fractions_varid,binary_lake_mask_varid
  integer, dimension(2) :: dimids
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
    call check_return_code(nf90_create(binary_lake_mask_filename,nf90_noclobber,ncid))
    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_def_dim(ncid, "lat",nlat_surface,lat_dimid))
    call check_return_code(nf90_def_dim(ncid, "lon",nlon_surface,lon_dimid))
    dimids = (/lon_dimid,lat_dimid/)
    _END_IF_USE_SINGLE_INDEX_
    call check_return_code(nf90_def_var(ncid, "number_fine_grid_cells", &
                                        nf90_int, dimids,number_fine_grid_cells_varid))
    call check_return_code(nf90_def_var(ncid, "lake_pixel_counts", &
                                        nf90_int, dimids,lake_pixel_counts_varid))
    call check_return_code(nf90_def_var(ncid, "lake_fractions", &
                                        nf90_real, dimids,lake_fractions_varid))
    call check_return_code(nf90_def_var(ncid, "binary_lake_mask", &
                                        nf90_real, dimids,binary_lake_mask_varid))
    call check_return_code(nf90_enddef(ncid))

    allocate(binary_lake_mask_int(_NPOINTS_SURFACE_))
    where (binary_lake_mask)
      binary_lake_mask_int(_DIMS_) = 1
    elsewhere
      binary_lake_mask_int(_DIMS_) = 0
    end where
    _IF_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,number_fine_grid_cells_varid,number_fine_grid_cells))
    call check_return_code(nf90_put_var(ncid,lake_pixel_counts_varid,lake_pixel_counts_field))
    call check_return_code(nf90_put_var(ncid,lake_fractions_varid,lake_fractions_field))
    call check_return_code(nf90_put_var(ncid,binary_lake_mask_varid,binary_lake_mask_int))
    _ELSE_IF_NOT_USE_LONLAT_
    call check_return_code(nf90_put_var(ncid,number_fine_grid_cells_varid,transpose(number_fine_grid_cells)))
    call check_return_code(nf90_put_var(ncid,lake_pixel_counts_varid,transpose(lake_pixel_counts_field)))
    call check_return_code(nf90_put_var(ncid,lake_fractions_varid,transpose(lake_fractions_field)))
    call check_return_code(nf90_put_var(ncid,binary_lake_mask_varid,transpose(binary_lake_mask_int)))
    _END_IF_USE_LONLAT_
    call check_return_code(nf90_close(ncid))
    deallocate(binary_lake_mask_int)
end subroutine write_binary_lake_mask_field

end module l2_lake_model_output
