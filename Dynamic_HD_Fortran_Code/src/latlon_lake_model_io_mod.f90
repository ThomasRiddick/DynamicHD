module latlon_lake_model_io_mod

use netcdf
use latlon_lake_model_mod
use check_return_code_netcdf_mod
use parameters_mod
implicit none

character(len = max_name_length) :: lake_params_filename
character(len = max_name_length) :: lake_start_filename
real :: lake_retention_coefficient

contains

subroutine config_lakes(lake_model_ctl_filename)
  include 'lake_model_ctl.inc'

  character(len = max_name_length) :: lake_model_ctl_filename
  integer :: unit_number

    open(newunit=unit_number,file=lake_model_ctl_filename,status='old')
    read (unit=unit_number,nml=lake_model_ctl)
    close(unit_number)

end subroutine config_lakes

function read_lake_parameters(instant_throughflow)&
                              result(lake_parameters)
  logical, intent(in) :: instant_throughflow
  type(lakeparameters), pointer :: lake_parameters
  logical, allocatable, dimension(:,:) :: lake_centers
  integer, allocatable, dimension(:,:) :: lake_centers_int
  real, allocatable, dimension(:,:) :: connection_volume_thresholds
  real, allocatable, dimension(:,:) :: flood_volume_thresholds
  logical, allocatable, dimension(:,:) :: flood_local_redirect
  integer, allocatable, dimension(:,:) :: flood_local_redirect_int
  logical, allocatable, dimension(:,:) :: connect_local_redirect
  integer, allocatable, dimension(:,:) :: connect_local_redirect_int
  logical, allocatable, dimension(:,:) :: additional_flood_local_redirect
  integer, allocatable, dimension(:,:) :: additional_flood_local_redirect_int
  logical, allocatable, dimension(:,:) :: additional_connect_local_redirect
  integer, allocatable, dimension(:,:) :: additional_connect_local_redirect_int
  integer, allocatable, dimension(:,:) :: merge_points
  integer, allocatable, dimension(:,:) :: flood_next_cell_lat_index
  integer, allocatable, dimension(:,:) :: flood_next_cell_lon_index
  integer, allocatable, dimension(:,:) :: connect_next_cell_lat_index
  integer, allocatable, dimension(:,:) :: connect_next_cell_lon_index
  integer, allocatable, dimension(:,:) :: flood_force_merge_lat_index
  integer, allocatable, dimension(:,:) :: flood_force_merge_lon_index
  integer, allocatable, dimension(:,:) :: connect_force_merge_lat_index
  integer, allocatable, dimension(:,:) :: connect_force_merge_lon_index
  integer, allocatable, dimension(:,:) :: flood_redirect_lat_index
  integer, allocatable, dimension(:,:) :: flood_redirect_lon_index
  integer, allocatable, dimension(:,:) :: connect_redirect_lat_index
  integer, allocatable, dimension(:,:) :: connect_redirect_lon_index
  integer, allocatable, dimension(:,:) :: additional_flood_redirect_lat_index
  integer, allocatable, dimension(:,:) :: additional_flood_redirect_lon_index
  integer, allocatable, dimension(:,:) :: additional_connect_redirect_lat_index
  integer, allocatable, dimension(:,:) :: additional_connect_redirect_lon_index
  integer :: nlat,nlon
  integer :: nlat_coarse,nlon_coarse
  integer :: ncid
  integer :: varid,dimid

    nlat_coarse = 360
    nlon_coarse = 720
    call check_return_code(nf90_open(lake_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'lat',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat))

    call check_return_code(nf90_inq_dimid(ncid,'lon',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon))

    call check_return_code(nf90_inq_varid(ncid,'lake_centers',varid))
    call check_return_code(nf90_get_var(ncid, varid,lake_centers_int))

    call check_return_code(nf90_inq_varid(ncid,'connection_volume_thresholds',varid))
    call check_return_code(nf90_get_var(ncid, varid,connection_volume_thresholds))

    call check_return_code(nf90_inq_varid(ncid,'flood_volume_thresholds',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_volume_thresholds))

    call check_return_code(nf90_inq_varid(ncid,'flood_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_local_redirect_int))

    call check_return_code(nf90_inq_varid(ncid,'connect_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_local_redirect_int))

    call check_return_code(nf90_inq_varid(ncid,'additional_flood_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_flood_local_redirect_int))

    call check_return_code(nf90_inq_varid(ncid,'additional_connect_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_connect_local_redirect_int))

    call check_return_code(nf90_inq_varid(ncid,'flood_next_cell_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_next_cell_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'flood_next_cell_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_next_cell_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'connect_next_cell_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_next_cell_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'connect_next_cell_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_next_cell_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'flood_force_merge_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,lake_centers_int))

    call check_return_code(nf90_inq_varid(ncid,'flood_force_merge_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_force_merge_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'connect_force_merge_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_force_merge_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'connect_force_merge_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_force_merge_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'flood_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_redirect_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'flood_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_redirect_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'connect_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_redirect_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'connect_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_redirect_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'additional_flood_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_flood_redirect_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'additional_flood_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_flood_redirect_lon_index))

    call check_return_code(nf90_inq_varid(ncid,'additional_connect_redirect_lat_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_connect_redirect_lat_index))

    call check_return_code(nf90_inq_varid(ncid,'additional_connect_redirect_lon_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_connect_redirect_lon_index))

    call check_return_code(nf90_close(ncid))

    where (lake_centers_int == 0)
      lake_centers = .false.
    elsewhere
      lake_centers = .true.
    end where

    where (flood_local_redirect_int == 0)
      flood_local_redirect = .false.
    elsewhere
      flood_local_redirect = .true.
    end where

    where (connect_local_redirect_int == 0)
      connect_local_redirect = .false.
    elsewhere
      connect_local_redirect = .true.
    end where

    where (additional_flood_local_redirect_int == 0)
      additional_flood_local_redirect = .false.
    elsewhere
      additional_flood_local_redirect = .true.
    end where

    where (additional_connect_local_redirect_int == 0)
      additional_connect_local_redirect = .false.
    elsewhere
      additional_connect_local_redirect = .true.
    end where

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

end function read_lake_parameters

subroutine load_lake_initial_values(initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers)
  real, allocatable, dimension(:,:), intent(out) :: initial_water_to_lake_centers
  real, allocatable, dimension(:,:), intent(out) :: initial_spillover_to_rivers
  integer :: nlat,nlon
  integer :: ncid,varid,dimid
    call check_return_code(nf90_open(lake_start_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'lat',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlat))

    call check_return_code(nf90_inq_dimid(ncid,'lon',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=nlon))

    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_lakes',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_water_to_lake_centers))

    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_rivers',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_spillover_to_rivers))

    call check_return_code(nf90_close(ncid))
end subroutine load_lake_initial_values

subroutine write_lake_volumes_field(lake_volumes_filename,&
                                  lake_parameters,lake_volumes)
  character(len = max_name_length) :: lake_volumes_filename
  real, allocatable, dimension(:,:), intent(inout) :: lake_volumes
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

! subroutine write_lake_numbers_field(lake_parameters,lake_fields,timestep)
!   type(lakeparameters), pointer, intent(in) :: lake_parameters
!   type(lakefields), pointer, intent(in) :: lake_fields
!   integer, intent(in) :: timestep
!   character(len = 50) :: timestep_str
!   character(len = max_name_length) :: filename
!   if(timestep == -1) then
!     filename = '/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_results.nc'
!   else
!     filename = '/Users/thomasriddick/Documents/data/temp/transient_sim_1/lake_model_results_'
!     write (timestep_str,*) timestep
!     filename = trim(filename) // timestep_str // '.nc'
!   end if
!   ! write_field(lake_parameters.grid,"lake_field",
!               ! lake_fields.lake_numbers,filename)
! end subroutine write_lake_numbers_field

end module latlon_lake_model_io_mod
