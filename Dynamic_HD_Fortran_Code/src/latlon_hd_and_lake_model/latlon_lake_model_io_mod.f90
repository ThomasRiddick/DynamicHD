module latlon_lake_model_io_mod

use netcdf
use latlon_lake_model_mod
use check_return_code_netcdf_mod
use parameters_mod
implicit none

character(len = max_name_length) :: lake_params_filename
character(len = max_name_length) :: lake_start_filename
real(dp) :: lake_retention_coefficient

contains

subroutine add_offset(array,offset,exceptions)
  integer, dimension(:,:), intent(inout) :: array
  integer, intent(in) :: offset
  integer, dimension(:), optional, intent(in) :: exceptions
  logical, dimension(:,:), allocatable :: exception_mask
  integer :: exception
  integer :: i
    allocate(exception_mask(size(array,1),size(array,2)))
    exception_mask = .False.
    if(present(exceptions)) then
      do i = 1,size(exceptions)
        exception = exceptions(i)
        where(array == exception)
          exception_mask=.True.
        end where
      end do
    end if
    where(.not. exception_mask)
      array = array + offset
    end where
end subroutine

subroutine config_lakes(lake_model_ctl_filename)
  include 'lake_model_ctl.inc'

  character(len = max_name_length) :: lake_model_ctl_filename
  integer :: unit_number

    write(*,*) "Reading namelist: " // trim(lake_model_ctl_filename)
    open(newunit=unit_number,file=lake_model_ctl_filename,status='old')
    read (unit=unit_number,nml=lake_model_ctl)
    close(unit_number)

end subroutine config_lakes

function read_lake_parameters(instant_throughflow)&
                              result(lake_parameters)
  logical, intent(in) :: instant_throughflow
  type(lakeparameters), pointer :: lake_parameters
  logical, pointer, dimension(:,:) :: lake_centers
  integer, pointer, dimension(:,:) :: lake_centers_int
  real(dp), pointer, dimension(:,:) :: connection_volume_thresholds
  real(dp), pointer, dimension(:,:) :: flood_volume_thresholds
  logical, pointer, dimension(:,:) :: flood_local_redirect
  integer, pointer, dimension(:,:) :: flood_local_redirect_int
  logical, pointer, dimension(:,:) :: connect_local_redirect
  integer, pointer, dimension(:,:) :: connect_local_redirect_int
  logical, pointer, dimension(:,:) :: additional_flood_local_redirect
  integer, pointer, dimension(:,:) :: additional_flood_local_redirect_int
  logical, pointer, dimension(:,:) :: additional_connect_local_redirect
  integer, pointer, dimension(:,:) :: additional_connect_local_redirect_int
  integer, pointer, dimension(:,:) :: merge_points
  integer, pointer, dimension(:,:) :: flood_next_cell_lat_index
  integer, pointer, dimension(:,:) :: flood_next_cell_lon_index
  integer, pointer, dimension(:,:) :: connect_next_cell_lat_index
  integer, pointer, dimension(:,:) :: connect_next_cell_lon_index
  integer, pointer, dimension(:,:) :: flood_force_merge_lat_index
  integer, pointer, dimension(:,:) :: flood_force_merge_lon_index
  integer, pointer, dimension(:,:) :: connect_force_merge_lat_index
  integer, pointer, dimension(:,:) :: connect_force_merge_lon_index
  integer, pointer, dimension(:,:) :: flood_redirect_lat_index
  integer, pointer, dimension(:,:) :: flood_redirect_lon_index
  integer, pointer, dimension(:,:) :: connect_redirect_lat_index
  integer, pointer, dimension(:,:) :: connect_redirect_lon_index
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lat_index
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lon_index
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lat_index
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lon_index
  integer, allocatable, dimension(:,:) :: temp_integer_array
  real(dp),    allocatable, dimension(:,:) :: temp_real_array
  integer :: nlat,nlon
  integer :: nlat_coarse,nlon_coarse
  integer :: ncid
  integer :: varid,dimid

    write(*,*) "Loading lake parameters from file: " // trim(lake_params_filename)
    nlat_coarse = 360
    nlon_coarse = 720

    call check_return_code(nf90_open(lake_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'lat',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat))

    call check_return_code(nf90_inq_dimid(ncid,'lon',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon))

    allocate(temp_real_array(nlon,nlat))
    allocate(temp_integer_array(nlon,nlat))

    call check_return_code(nf90_inq_varid(ncid,'lake_centers',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(lake_centers_int(nlat,nlon))
    lake_centers_int = transpose(temp_integer_array)

    call check_return_code(nf90_inq_varid(ncid,'connection_volume_thresholds',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    allocate(connection_volume_thresholds(nlat,nlon))
    connection_volume_thresholds = transpose(temp_real_array)

    call check_return_code(nf90_inq_varid(ncid,'flood_volume_thresholds',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    allocate(flood_volume_thresholds(nlat,nlon))
    flood_volume_thresholds = transpose(temp_real_array)

    call check_return_code(nf90_inq_varid(ncid,'flood_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_local_redirect_int(nlat,nlon))
    flood_local_redirect_int = transpose(temp_integer_array)

    call check_return_code(nf90_inq_varid(ncid,'connect_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_local_redirect_int(nlat,nlon))
    connect_local_redirect_int = transpose(temp_integer_array)

    call check_return_code(nf90_inq_varid(ncid,'additional_flood_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(additional_flood_local_redirect_int(nlat,nlon))
    additional_flood_local_redirect_int = transpose(temp_integer_array)

    call check_return_code(nf90_inq_varid(ncid,'additional_connect_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(additional_connect_local_redirect_int(nlat,nlon))
    additional_connect_local_redirect_int = transpose(temp_integer_array)

    call check_return_code(nf90_inq_varid(ncid,'flood_next_cell_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_next_cell_lat_index(nlat,nlon))
    flood_next_cell_lat_index = transpose(temp_integer_array)
    call add_offset(flood_next_cell_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'flood_next_cell_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_next_cell_lon_index(nlat,nlon))
    flood_next_cell_lon_index = transpose(temp_integer_array)
    call add_offset(flood_next_cell_lon_index ,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'connect_next_cell_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_next_cell_lat_index(nlat,nlon))
    connect_next_cell_lat_index = transpose(temp_integer_array)
    call add_offset(connect_next_cell_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'connect_next_cell_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_next_cell_lon_index(nlat,nlon))
    connect_next_cell_lon_index = transpose(temp_integer_array)
    call add_offset(connect_next_cell_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'flood_force_merge_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_force_merge_lat_index(nlat,nlon))
    flood_force_merge_lat_index = transpose(temp_integer_array)
    call add_offset(flood_force_merge_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'flood_force_merge_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_force_merge_lon_index(nlat,nlon))
    flood_force_merge_lon_index = transpose(temp_integer_array)
    call add_offset(flood_force_merge_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'connect_force_merge_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_force_merge_lat_index(nlat,nlon))
    connect_force_merge_lat_index = transpose(temp_integer_array)
    call add_offset(connect_force_merge_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'connect_force_merge_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_force_merge_lon_index(nlat,nlon))
    connect_force_merge_lon_index = transpose(temp_integer_array)
    call add_offset(connect_force_merge_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'flood_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_redirect_lat_index(nlat,nlon))
    flood_redirect_lat_index = transpose(temp_integer_array)
    call add_offset(flood_redirect_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'flood_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(flood_redirect_lon_index(nlat,nlon))
    flood_redirect_lon_index = transpose(temp_integer_array)
    call add_offset(flood_redirect_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'connect_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_redirect_lat_index(nlat,nlon))
    connect_redirect_lat_index = transpose(temp_integer_array)
    call add_offset(connect_redirect_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'connect_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(connect_redirect_lon_index(nlat,nlon))
    connect_redirect_lon_index = transpose(temp_integer_array)
    call add_offset(connect_redirect_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'additional_flood_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(additional_flood_redirect_lat_index(nlat,nlon))
    additional_flood_redirect_lat_index = transpose(temp_integer_array)
    call add_offset(additional_flood_redirect_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'additional_flood_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(additional_flood_redirect_lon_index(nlat,nlon))
    additional_flood_redirect_lon_index  = transpose(temp_integer_array)
    call add_offset(additional_flood_redirect_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'additional_connect_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(additional_connect_redirect_lat_index(nlat,nlon))
    additional_connect_redirect_lat_index = transpose(temp_integer_array)
    call add_offset(additional_connect_redirect_lat_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'additional_connect_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(additional_connect_redirect_lon_index(nlat,nlon))
    additional_connect_redirect_lon_index  = transpose(temp_integer_array)
    call add_offset(additional_connect_redirect_lon_index,1,(/-1/))

    call check_return_code(nf90_inq_varid(ncid,'merge_points',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    allocate(merge_points(nlat,nlon))
    merge_points  = transpose(temp_integer_array)

    call check_return_code(nf90_close(ncid))

    allocate(lake_centers(nlat,nlon))
    where (lake_centers_int == 0)
      lake_centers = .false.
    elsewhere
      lake_centers = .true.
    end where

    allocate(flood_local_redirect(nlat,nlon))
    where (flood_local_redirect_int == 0)
      flood_local_redirect = .false.
    elsewhere
      flood_local_redirect = .true.
    end where

    allocate(connect_local_redirect(nlat,nlon))
    where (connect_local_redirect_int == 0)
      connect_local_redirect = .false.
    elsewhere
      connect_local_redirect = .true.
    end where

    allocate(additional_flood_local_redirect(nlat,nlon))
    where (additional_flood_local_redirect_int == 0)
      additional_flood_local_redirect = .false.
    elsewhere
      additional_flood_local_redirect = .true.
    end where

    allocate(additional_connect_local_redirect(nlat,nlon))
    where (additional_connect_local_redirect_int == 0)
      additional_connect_local_redirect = .false.
    elsewhere
      additional_connect_local_redirect = .true.
    end where
    deallocate(lake_centers_int)
    deallocate(flood_local_redirect_int)
    deallocate(connect_local_redirect_int)
    deallocate(additional_flood_local_redirect_int)
    deallocate(additional_connect_local_redirect_int)
    lake_parameters => lakeparameters(lake_centers, &
                                      connection_volume_thresholds, &
                                      flood_volume_thresholds, &
                                      flood_local_redirect, &
                                      connect_local_redirect, &
                                      additional_flood_local_redirect, &
                                      additional_connect_local_redirect, &
                                      merge_points, &
                                      flood_next_cell_lat_index, &
                                      flood_next_cell_lon_index, &
                                      connect_next_cell_lat_index, &
                                      connect_next_cell_lon_index, &
                                      flood_force_merge_lat_index, &
                                      flood_force_merge_lon_index, &
                                      connect_force_merge_lat_index, &
                                      connect_force_merge_lon_index, &
                                      flood_redirect_lat_index, &
                                      flood_redirect_lon_index, &
                                      connect_redirect_lat_index, &
                                      connect_redirect_lon_index, &
                                      additional_flood_redirect_lat_index, &
                                      additional_flood_redirect_lon_index, &
                                      additional_connect_redirect_lat_index, &
                                      additional_connect_redirect_lon_index, &
                                      nlat,nlon, &
                                      nlat_coarse,nlon_coarse, &
                                      instant_throughflow, &
                                      lake_retention_coefficient)
    deallocate(temp_integer_array)
    deallocate(temp_real_array)
end function read_lake_parameters

subroutine load_lake_initial_values(initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers, &
                                    step_length)
  real(dp), pointer, dimension(:,:), intent(inout) :: initial_water_to_lake_centers
  real(dp), pointer, dimension(:,:), intent(inout) :: initial_spillover_to_rivers
  real(dp), pointer, dimension(:,:) :: initial_water_to_lake_centers_temp
  real(dp), pointer, dimension(:,:) :: initial_spillover_to_rivers_temp
  real(dp) :: step_length
  integer :: nlat_coarse,nlon_coarse
  integer :: nlat,nlon
  integer :: ncid,varid,dimid

    nlat_coarse = 360
    nlon_coarse = 720
    write(*,*) "Loading lake initial values from file: " // trim(lake_start_filename)
    call check_return_code(nf90_open(lake_start_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'lat',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat))

    call check_return_code(nf90_inq_dimid(ncid,'lon',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon))

    allocate(initial_water_to_lake_centers_temp(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_lakes',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_water_to_lake_centers_temp))
    allocate(initial_water_to_lake_centers(nlat,nlon))
    initial_water_to_lake_centers = transpose(initial_water_to_lake_centers_temp)

    allocate(initial_spillover_to_rivers_temp(nlon_coarse,nlat_coarse))
    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_rivers',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_spillover_to_rivers_temp))
    allocate(initial_spillover_to_rivers(nlat_coarse,nlon_coarse))
    initial_spillover_to_rivers = transpose(initial_spillover_to_rivers_temp)
    initial_spillover_to_rivers(:,:) = initial_spillover_to_rivers(:,:)/step_length
    call check_return_code(nf90_close(ncid))
    deallocate(initial_water_to_lake_centers_temp)
    deallocate(initial_spillover_to_rivers_temp)
end subroutine load_lake_initial_values

subroutine write_lake_volumes_field(lake_volumes_filename,&
                                  lake_parameters,lake_volumes)
  character(len = max_name_length) :: lake_volumes_filename
  real(dp), pointer, dimension(:,:), intent(inout) :: lake_volumes
  type(lakeparameters), pointer :: lake_parameters
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
    call check_return_code(nf90_create(lake_volumes_filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid, "lat",lake_parameters%nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid, "lon",lake_parameters%nlon,lon_dimid))
    dimids = (/lat_dimid,lon_dimid/)
    call check_return_code(nf90_def_var(ncid, "lake_field", nf90_real, dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,lake_volumes))
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_volumes_field

subroutine write_lake_numbers_field(lake_parameters,lake_fields,timestep)
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakefields), pointer, intent(in) :: lake_fields
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
    if(timestep == -1) then
      filename = 'lake_model_results.nc'
    else
      filename = 'lake_model_results_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"lat",lake_parameters%nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",lake_parameters%nlon,lon_dimid))
    ! call check_return_code(nf90_def_var(ncid,"lat",nf90_double,lat_dimid,lat_varid))
    ! call check_return_code(nf90_def_var(ncid,"lon",nf90_double,lon_dimid,lon_varid))
    dimids = (/lat_dimid,lon_dimid/)
    call check_return_code(nf90_def_var(ncid,"lake_number",nf90_real,dimids,varid))
    ! call check_return_code(nf90_put_var(ncid,lat_varid,))
    ! call check_return_code(nf90_put_var(ncid,lon_varid,))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,&
                                        lake_fields%lake_numbers))
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_numbers_field

subroutine write_diagnostic_lake_volumes(lake_parameters, &
                                         lake_prognostics, &
                                         lake_fields, &
                                         timestep)
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakeprognostics),pointer, intent(in) :: lake_prognostics
  type(lakefields), pointer, intent(in) :: lake_fields
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
  integer :: ncid,varid,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volume
    diagnostic_lake_volume => calculate_diagnostic_lake_volumes(lake_parameters,&
                                                                lake_prognostics,&
                                                                lake_fields)
    if(timestep == -1) then
      filename = 'diagnostic_lake_volume_results.nc'
    else
      filename = 'diagnostic_lake_volume_results_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"lat",lake_parameters%nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",lake_parameters%nlon,lon_dimid))
    dimids = (/lat_dimid,lon_dimid/)
    call check_return_code(nf90_def_var(ncid,"diagnostic_lake_volume",nf90_real,dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,diagnostic_lake_volume))
    call check_return_code(nf90_close(ncid))
    deallocate(diagnostic_lake_volume)
end subroutine write_diagnostic_lake_volumes

end module latlon_lake_model_io_mod
