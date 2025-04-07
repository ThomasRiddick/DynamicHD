module l2_lake_model_input

use netcdf
use l2_lake_model_mod
use check_return_code_netcdf_mod
use parameters_mod

implicit none

character(len = max_name_length) :: lake_para_filename
real(dp) :: lake_retention_coefficient
real(dp) :: minimum_lake_volume_threshold
logical :: run_water_budget_check

contains

subroutine set_lake_parameters_filename(lake_para_filename_in)
    character(len = max_name_length) :: lake_para_filename_in
      lake_para_filename = lake_para_filename_in
end subroutine set_lake_parameters_filename

subroutine load_lake_parameters(cell_areas_on_surface_model_grid, &
                                load_binary_mask, &
                                lake_model_parameters, &
                                lake_parameters_as_array, &
                                _NPOINTS_HD_, &
                                _NPOINTS_SURFACE_)
  real(dp), dimension(_DIMS_), pointer, intent(in) :: cell_areas_on_surface_model_grid
  logical, intent(in) :: load_binary_mask
  type(lakemodelparameters), pointer, intent(out) :: lake_model_parameters
  real(dp), dimension(:), pointer, intent(out) :: lake_parameters_as_array
  _DEF_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_
  integer, dimension(_DIMS_), pointer :: temp_integer_array
  integer, dimension(_DIMS_), pointer :: binary_lake_mask_int_temp_array
  integer, dimension(:), pointer :: number_of_lakes_temp_array
  integer, dimension(_DIMS_), pointer :: is_lake_int
  integer, dimension(_DIMS_), pointer :: binary_lake_mask_int
  logical, dimension(_DIMS_), pointer :: is_lake
  logical, dimension(_DIMS_), pointer :: binary_lake_mask
  integer :: number_of_lakes
  integer :: npoints
  _DEF_NPOINTS_HD_ _INTENT_in_
  _DEF_NPOINTS_LAKE_
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
  integer :: ncid,dimid,varid
    write(*,*) "Loading: " // trim(lake_para_filename)

    call check_return_code(nf90_open(lake_para_filename,nf90_nowrite,ncid))

    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_inq_dimid(ncid,'latitude',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat_lake))

    call check_return_code(nf90_inq_dimid(ncid,'longitude',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon_lake))

    _END_IF_USE_SINGLE_INDEX_

    call check_return_code(nf90_inq_dimid(ncid,'npoints',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=npoints))

    allocate(temp_integer_array(_NPOINTS_LAKE_))
    allocate(binary_lake_mask_int_temp_array(_NPOINTS_SURFACE_))
    allocate(number_of_lakes_temp_array(1))
    allocate(lake_parameters_as_array(npoints))

    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_inq_varid(ncid,'corresponding_surface_cell_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(corresponding_surface_cell_lat_index(_NPOINTS_LAKE_))
    corresponding_surface_cell_lat_index  = transpose(temp_integer_array)

    call check_return_code(nf90_inq_varid(ncid,'corresponding_surface_cell_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(corresponding_surface_cell_lon_index(_NPOINTS_LAKE_))
    corresponding_surface_cell_lon_index  = transpose(temp_integer_array)
    _END_IF_USE_SINGLE_INDEX_

    call check_return_code(nf90_inq_varid(ncid,'number_of_lakes',varid))
    call check_return_code(nf90_get_var(ncid, varid,number_of_lakes_temp_array))
    number_of_lakes = number_of_lakes_temp_array(1)

    call check_return_code(nf90_inq_varid(ncid,'lake_mask',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(is_lake_int(_NPOINTS_LAKE_))
    is_lake_int = transpose(temp_integer_array)

    allocate(binary_lake_mask_int(_NPOINTS_SURFACE_))
    if (load_binary_mask) then
      call check_return_code(nf90_inq_varid(ncid,'binary_lake_mask',varid))
      call check_return_code(nf90_get_var(ncid, varid,binary_lake_mask_int_temp_array))
      binary_lake_mask_int  = transpose(binary_lake_mask_int_temp_array)
    else
      binary_lake_mask_int(_DIMS_) = 0
    end if

    call check_return_code(nf90_inq_varid(ncid,'lakes_as_array',varid))
    call check_return_code(nf90_get_var(ncid, varid,lake_parameters_as_array))

    call check_return_code(nf90_close(ncid))

    allocate(is_lake(_NPOINTS_LAKE_))
    where (is_lake_int == 1)
      is_lake = .true.
    elsewhere
      is_lake = .false.
    end where

    allocate(binary_lake_mask(_NPOINTS_SURFACE_))
    where (binary_lake_mask_int == 1)
      binary_lake_mask = .true.
    elsewhere
      binary_lake_mask = .false.
    end where

    deallocate(is_lake_int)
    deallocate(binary_lake_mask_int)
    deallocate(temp_integer_array)
    deallocate(binary_lake_mask_int_temp_array)
    deallocate(number_of_lakes_temp_array)

    lake_model_parameters => &
      lakemodelparameters(_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                          cell_areas_on_surface_model_grid, &
                          number_of_lakes, &
                          is_lake, &
                          binary_lake_mask, &
                          _NPOINTS_HD_, &
                          _NPOINTS_LAKE_, &
                          _NPOINTS_SURFACE_, &
                          lake_retention_coefficient, &
                          minimum_lake_volume_threshold)
    deallocate(is_lake)
end subroutine load_lake_parameters

end module l2_lake_model_input
