module lake_model_input

use netcdf
use lake_model_mod
use check_return_code_netcdf_mod
use parameters_mod

implicit none

character(len = max_name_length) :: lake_params_filename
character(len = max_name_length) :: lake_start_filename
real(dp) :: lake_retention_coefficient
real(dp) :: minimum_lake_volume_threshold
logical :: run_water_budget_check
logical :: use_binary_lake_mask

contains

subroutine config_lakes(lake_model_ctl_filename,&
                        run_water_budget_check_out, &
                        use_binary_lake_mask_out)
  include 'lake_model_ctl.inc'
  character(len = max_name_length) :: lake_model_ctl_filename
  logical, intent(out) :: run_water_budget_check_out
  logical, intent(out) :: use_binary_lake_mask_out
  integer :: unit_number
    write(*,*) "Reading namelist: " // trim(lake_model_ctl_filename)
    open(newunit=unit_number,file=lake_model_ctl_filename,status='old')
    read(unit=unit_number,nml=lake_model_ctl)
    close(unit_number)
    run_water_budget_check_out = run_water_budget_check
    use_binary_lake_mask_out = use_binary_lake_mask
end subroutine config_lakes

subroutine set_lake_parameters_filename(lake_params_filename_in)
    character(len = max_name_length) :: lake_params_filename_in
      lake_params_filename = lake_params_filename_in
end subroutine set_lake_parameters_filename

subroutine load_lake_model_parameters(cell_areas_on_surface_model_grid, &
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
  real(dp), dimension(_DIMS_), pointer :: raw_orography
  real(dp), dimension(_DIMS_), pointer :: temp_real_array
  integer, dimension(_DIMS_), pointer :: temp_integer_array
  integer, dimension(_DIMS_), pointer :: non_lake_mask_int_temp_array
  integer, dimension(_DIMS_), pointer :: binary_lake_mask_int_temp_array
  integer, dimension(:), pointer :: number_of_lakes_temp_array
  integer, dimension(_DIMS_), pointer :: is_lake_int
  integer, dimension(_DIMS_), pointer :: non_lake_mask_int
  integer, dimension(_DIMS_), pointer :: binary_lake_mask_int
  logical, dimension(_DIMS_), pointer :: is_lake
  logical, dimension(_DIMS_), pointer :: non_lake_mask
  logical, dimension(_DIMS_), pointer :: binary_lake_mask
  integer :: number_of_lakes
  integer :: npoints
  _DEF_NPOINTS_HD_ _INTENT_in_
  _DEF_NPOINTS_LAKE_
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
  integer :: ncid,dimid,varid
    write(*,*) "Loading: " // trim(lake_params_filename)

    call check_return_code(nf90_open(lake_params_filename,nf90_nowrite,ncid))

    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_inq_dimid(ncid,'lat',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat_lake))

    call check_return_code(nf90_inq_dimid(ncid,'lon',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon_lake))

    _END_IF_USE_SINGLE_INDEX_

    call check_return_code(nf90_inq_dimid(ncid,'npoints',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=npoints))

    allocate(temp_integer_array(nlon_lake,nlat_lake))
    allocate(temp_real_array(nlon_lake,nlat_lake))
    allocate(binary_lake_mask_int_temp_array(nlon_surface,nlat_surface))
    allocate(non_lake_mask_int_temp_array(nlon_surface,nlat_surface))
    allocate(number_of_lakes_temp_array(1))
    allocate(lake_parameters_as_array(npoints))

    _IF_USE_SINGLE_INDEX_
     ! Not implemented
    _ELSE_
    call check_return_code(nf90_inq_varid(ncid,'corresponding_surface_cell_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(corresponding_surface_cell_lat_index(_NPOINTS_LAKE_))
    _IF_USE_LONLAT_
    corresponding_surface_cell_lat_index  = temp_integer_array
    _ELSE_IF_NOT_USE_LONLAT_
    corresponding_surface_cell_lat_index  = transpose(temp_integer_array)
    _END_IF_USE_LONLAT_

    call check_return_code(nf90_inq_varid(ncid,'corresponding_surface_cell_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(corresponding_surface_cell_lon_index(_NPOINTS_LAKE_))
    _IF_USE_LONLAT_
    corresponding_surface_cell_lon_index  = temp_integer_array
    _ELSE_IF_NOT_USE_LONLAT_
    corresponding_surface_cell_lon_index  = transpose(temp_integer_array)
    _END_IF_USE_LONLAT_
    _END_IF_USE_SINGLE_INDEX_

    call check_return_code(nf90_inq_varid(ncid,'number_of_lakes',varid))
    call check_return_code(nf90_get_var(ncid, varid,number_of_lakes_temp_array))
    number_of_lakes = number_of_lakes_temp_array(1)

    call check_return_code(nf90_inq_varid(ncid,'lake_mask',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(is_lake_int(_NPOINTS_LAKE_))
    _IF_USE_LONLAT_
    is_lake_int = temp_integer_array
    _ELSE_IF_NOT_USE_LONLAT_
    is_lake_int = transpose(temp_integer_array)
    _END_IF_USE_LONLAT_

    call check_return_code(nf90_inq_varid(ncid,'raw_orography',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    allocate(raw_orography(_NPOINTS_LAKE_))
    _IF_USE_LONLAT_
    raw_orography = temp_real_array
    _ELSE_IF_NOT_USE_LONLAT_
    raw_orography = transpose(temp_real_array)
    _END_IF_USE_LONLAT_

    allocate(binary_lake_mask_int(_NPOINTS_SURFACE_))
    if (load_binary_mask) then
      call check_return_code(nf90_inq_varid(ncid,'binary_lake_mask',varid))
      call check_return_code(nf90_get_var(ncid, varid,binary_lake_mask_int_temp_array))
      _IF_USE_LONLAT_
      binary_lake_mask_int  = binary_lake_mask_int_temp_array
      _ELSE_IF_NOT_USE_LONLAT_
      binary_lake_mask_int  = transpose(binary_lake_mask_int_temp_array)
      _END_IF_USE_LONLAT_
    else
      binary_lake_mask_int(_DIMS_) = 0
    end if

    allocate(non_lake_mask_int(_NPOINTS_SURFACE_))
    call check_return_code(nf90_inq_varid(ncid,'non_lake_mask',varid))
    call check_return_code(nf90_get_var(ncid, varid,non_lake_mask_int_temp_array))
    _IF_USE_LONLAT_
    non_lake_mask_int  = non_lake_mask_int_temp_array
    _ELSE_IF_NOT_USE_LONLAT_
    non_lake_mask_int  = transpose(non_lake_mask_int_temp_array)
    _END_IF_USE_LONLAT_

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

    allocate(non_lake_mask(_NPOINTS_SURFACE_))
    where (non_lake_mask_int == 1)
      non_lake_mask = .true.
    elsewhere
      non_lake_mask = .false.
    end where

    deallocate(is_lake_int)
    deallocate(binary_lake_mask_int)
    deallocate(non_lake_mask_int)
    deallocate(temp_integer_array)
    deallocate(binary_lake_mask_int_temp_array)
    deallocate(non_lake_mask_int_temp_array)
    deallocate(number_of_lakes_temp_array)

    lake_model_parameters => &
      lakemodelparameters(_INDICES_FIELD_corresponding_surface_cell_INDEX_NAME_index_, &
                          cell_areas_on_surface_model_grid, &
                          number_of_lakes, &
                          is_lake, &
                          raw_orography, &
                          non_lake_mask, &
                          binary_lake_mask, &
                          _NPOINTS_HD_, &
                          _NPOINTS_LAKE_, &
                          _NPOINTS_SURFACE_, &
                          lake_retention_coefficient, &
                          minimum_lake_volume_threshold)
    deallocate(is_lake)
end subroutine load_lake_model_parameters

subroutine load_lake_initial_values(initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers, &
                                    step_length, &
                                    _NPOINTS_HD_)
  real(dp), pointer, dimension(_DIMS_), intent(inout) :: initial_water_to_lake_centers
  real(dp), pointer, dimension(_DIMS_), intent(inout) :: initial_spillover_to_rivers
  real(dp), pointer, dimension(_DIMS_) :: initial_water_to_lake_centers_temp
  real(dp), pointer, dimension(_DIMS_) :: initial_spillover_to_rivers_temp
  real(dp) :: step_length
  integer :: ncid,varid,dimid
  _DEF_NPOINTS_HD_ _INTENT_in_
  _DEF_NPOINTS_LAKE_

    write(*,*) "Loading lake initial values from file: " // trim(lake_start_filename)
    call check_return_code(nf90_open(lake_start_filename,nf90_nowrite,ncid))

    _IF_USE_SINGLE_INDEX_
    ! Not implemented
    _ELSE_
    call check_return_code(nf90_inq_dimid(ncid,'lat',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat_lake))

    call check_return_code(nf90_inq_dimid(ncid,'lon',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon_lake))
    _END_IF_USE_SINGLE_INDEX_

    allocate(initial_water_to_lake_centers_temp(nlon_lake,nlat_lake))
    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_lakes',varid))
    call check_return_code(nf90_get_var(ncid, varid, &
                                        initial_water_to_lake_centers_temp))
    allocate(initial_water_to_lake_centers(_NPOINTS_LAKE_))
    _IF_USE_LONLAT_
    initial_water_to_lake_centers = initial_water_to_lake_centers_temp
    _ELSE_IF_NOT_USE_LONLAT_
    initial_water_to_lake_centers = transpose(initial_water_to_lake_centers_temp)
    _END_IF_USE_LONLAT_

    allocate(initial_spillover_to_rivers_temp(nlon_hd,nlat_hd))
    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_rivers',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_spillover_to_rivers_temp))
    allocate(initial_spillover_to_rivers(_NPOINTS_HD_))
    _IF_USE_LONLAT_
    initial_spillover_to_rivers = initial_spillover_to_rivers_temp
    _ELSE_IF_NOT_USE_LONLAT_
    initial_spillover_to_rivers = transpose(initial_spillover_to_rivers_temp)
    _END_IF_USE_LONLAT_
    initial_spillover_to_rivers(_DIMS_) = initial_spillover_to_rivers(_DIMS_)/step_length
    call check_return_code(nf90_close(ncid))
    deallocate(initial_water_to_lake_centers_temp)
    deallocate(initial_spillover_to_rivers_temp)
end subroutine load_lake_initial_values

function load_cell_areas_on_surface_model_grid(surface_cell_areas_filename, &
                                               _NPOINTS_SURFACE_) &
    result(cell_areas_on_surface_model_grid)
  character(len = max_name_length) :: surface_cell_areas_filename
  real(dp), dimension(_DIMS_), pointer :: cell_areas_on_surface_model_grid
  real(dp), dimension(_DIMS_), pointer :: cell_areas_on_surface_model_grid_temp
  integer :: ncid,varid
  _DEF_NPOINTS_SURFACE_ _INTENT_in_

    write(*,*) "Loading surface grid cell areas from file: " // trim(surface_cell_areas_filename)
    call check_return_code(nf90_open(surface_cell_areas_filename,nf90_nowrite,ncid))

    allocate(cell_areas_on_surface_model_grid_temp(nlon_surface,nlat_surface))
    call check_return_code(nf90_inq_varid(ncid,'cell_areas_on_surface_model_grid',varid))
    call check_return_code(nf90_get_var(ncid, varid, &
                                        cell_areas_on_surface_model_grid_temp))
    allocate(cell_areas_on_surface_model_grid(_NPOINTS_SURFACE_))
    _IF_USE_LONLAT_
    cell_areas_on_surface_model_grid = cell_areas_on_surface_model_grid_temp
    _ELSE_IF_NOT_USE_LONLAT_
    cell_areas_on_surface_model_grid = transpose(cell_areas_on_surface_model_grid_temp)
    _END_IF_USE_LONLAT_

    call check_return_code(nf90_close(ncid))
    deallocate(cell_areas_on_surface_model_grid_temp)

end function load_cell_areas_on_surface_model_grid

end module lake_model_input
