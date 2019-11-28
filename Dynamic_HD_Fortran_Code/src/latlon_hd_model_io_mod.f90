module latlon_hd_model_io_mod

use netcdf
use latlon_lake_model_mod
use latlon_hd_model_mod
implicit none

integer, parameter :: max_name_length = 500

character(len = max_name_length) :: lake_params_filename
character(len = max_name_length) :: lake_init_cons_filename
real :: lake_retention_constant

contains

subroutine check_return_code(return_code)
  integer, intent(in) :: return_code
  if(return_code /= nf90_noerr) then
    print *,trim(nf90_strerror(return_code))
    stop
  end if
end subroutine check_return_code

subroutine config_lakes(lake_model_ctl_filename)
  include 'lake_model_ctl.inc'

  character(len = max_name_length) :: lake_model_ctl_filename
  integer :: unit_number

    open(newunit=unit_number,file=lake_model_ctl_filename,status='old')
    read (unit=unit_number,nml=lake_model_ctl)
    close(unit_number)

end subroutine config_lakes

function read_lake_parameters(lake_params_filename,instant_throughflow,&
                              lake_retention_coefficient) &
                              result(lake_parameters)
  character(len = max_name_length) :: lake_params_filename
  logical, intent(in) :: instant_throughflow
  real, intent(in) :: lake_retention_coefficient
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

function read_river_parameters(river_params_filename) &
    result(river_parameters)
  character(len = max_name_length) :: river_params_filename
  type(riverparameters), pointer :: river_parameters
  real, allocatable, dimension(:,:) :: rdirs
  integer, allocatable, dimension(:,:) :: river_reservoir_nums
  integer, allocatable, dimension(:,:) :: overland_reservoir_nums
  integer, allocatable, dimension(:,:) :: base_reservoir_nums
  real, allocatable, dimension(:,:) :: river_retention_coefficients
  real, allocatable, dimension(:,:) :: overland_retention_coefficients
  real, allocatable, dimension(:,:) :: base_retention_coefficients
  integer, allocatable, dimension(:,:) :: landsea_mask_int
  logical, allocatable, dimension(:,:) :: landsea_mask
  integer :: ncid,varid

    call check_return_code(nf90_open(river_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,'rdirs',varid))
    call check_return_code(nf90_get_var(ncid, varid,rdirs))

    call check_return_code(nf90_inq_varid(ncid,'river_reservoir_nums',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_reservoir_nums))

    call check_return_code(nf90_inq_varid(ncid,'base_reservoir_nums',varid))
    call check_return_code(nf90_get_var(ncid, varid,base_reservoir_nums))

    call check_return_code(nf90_inq_varid(ncid,'river_retention_coefficients',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_retention_coefficients))

    call check_return_code(nf90_inq_varid(ncid,'overland_retention_coefficients',varid))
    call check_return_code(nf90_get_var(ncid, varid,overland_retention_coefficients))

    call check_return_code(nf90_inq_varid(ncid,'base_retention_coefficients',varid))
    call check_return_code(nf90_get_var(ncid, varid,base_retention_coefficients))

    call check_return_code(nf90_inq_varid(ncid,'landsea_mask',varid))
    call check_return_code(nf90_get_var(ncid,varid,landsea_mask_int))

    where (landsea_mask_int == 0)
      landsea_mask = .false.
    elsewhere
      landsea_mask = .true.
    end where

    call check_return_code(nf90_close(ncid))

    allocate(river_parameters, &
             source= riverparameters(rdirs,river_reservoir_nums, &
                                     overland_reservoir_nums, &
                                     base_reservoir_nums, &
                                     river_retention_coefficients, &
                                     overland_retention_coefficients, &
                                     base_retention_coefficients, &
                                     landsea_mask))

end function read_river_parameters

function load_river_initial_values(hd_start_filename) &
    result(river_prognostic_fields)
  character(len = max_name_length) :: hd_start_filename
  type(riverprognosticfields), pointer :: river_prognostic_fields
  real, allocatable, dimension(:,:) :: river_inflow
  real, allocatable, dimension(:,:,:) :: base_flow_reservoirs
  real, allocatable, dimension(:,:) :: base_flow_reservoirs_temp
  real, allocatable, dimension(:,:,:) :: overland_flow_reservoirs
  real, allocatable, dimension(:,:) :: overland_flow_reservoirs_temp
  real, allocatable, dimension(:,:,:) :: river_flow_reservoirs
  real, allocatable, dimension(:,:) :: river_flow_reservoirs_temp1
  real, allocatable, dimension(:,:) :: river_flow_reservoirs_temp2
  real, allocatable, dimension(:,:) :: river_flow_reservoirs_temp3
  real, allocatable, dimension(:,:) :: river_flow_reservoirs_temp4
  real, allocatable, dimension(:,:) :: river_flow_reservoirs_temp5
  integer :: nlat,nlon
  integer :: ncid,varid
    nlat = 360
    nlon = 720

    call check_return_code(nf90_open(hd_start_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,'FINFL',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_inflow))

    call check_return_code(nf90_inq_varid(ncid,'FGMEM',varid))
    call check_return_code(nf90_get_var(ncid, varid,base_flow_reservoirs_temp))

    call check_return_code(nf90_inq_varid(ncid,'FLFMEM',varid))
    call check_return_code(nf90_get_var(ncid, varid,overland_flow_reservoirs_temp))

    call check_return_code(nf90_inq_varid(ncid,'FRFMEM1',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp1))

    call check_return_code(nf90_inq_varid(ncid,'FRFMEM2',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp2))

    call check_return_code(nf90_inq_varid(ncid,'FRFMEM3',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp3))

    call check_return_code(nf90_inq_varid(ncid,'FRFMEM4',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp4))

    call check_return_code(nf90_inq_varid(ncid,'FRFMEM5',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp5))

    call check_return_code(nf90_close(ncid))

    allocate(base_flow_reservoirs(nlat,nlon,1))
    allocate(overland_flow_reservoirs(nlat,nlon,1))
    allocate(river_flow_reservoirs(nlat,nlon,5))
    base_flow_reservoirs(:,:,1) = base_flow_reservoirs_temp(:,:)
    overland_flow_reservoirs(:,:,1) = overland_flow_reservoirs_temp(:,:)
    river_flow_reservoirs(:,:,1) = river_flow_reservoirs_temp1(:,:)
    river_flow_reservoirs(:,:,2) = river_flow_reservoirs_temp2(:,:)
    river_flow_reservoirs(:,:,3) = river_flow_reservoirs_temp3(:,:)
    river_flow_reservoirs(:,:,4) = river_flow_reservoirs_temp4(:,:)
    river_flow_reservoirs(:,:,5) = river_flow_reservoirs_temp5(:,:)


    allocate(river_prognostic_fields,source=riverprognosticfields(river_inflow, &
                                                                  base_flow_reservoirs, &
                                                                  overland_flow_reservoirs, &
                                                                  river_flow_reservoirs))

end function load_river_initial_values

subroutine load_lake_initial_values(lake_start_filename,&
                                    initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers)
  character(len = max_name_length) :: lake_start_filename
  real, allocatable, dimension(:,:), intent(inout) :: initial_water_to_lake_centers
  real, allocatable, dimension(:,:), intent(inout) :: initial_spillover_to_rivers
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

subroutine write_river_initial_values(hd_start_filename,river_parameters, &
                                      river_prognostic_fields)
  character(len = max_name_length) :: hd_start_filename
  type(lakeparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_prognostic_fields
  integer :: ncid,varid_finfl,varid_fgmem
  integer :: varid_flfmem,varid_frfmem,lat_dimid,lon_dimid
  integer, dimension(2) :: dimids
    call check_return_code(nf90_create(hd_start_filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"lat",river_parameters%nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",river_parameters%nlon,lon_dimid))
    dimids = (/lat_dimid,lon_dimid/)
    call check_return_code(nf90_def_var(ncid,"FINFL", nf90_real,dimids,varid_finfl))
    call check_return_code(nf90_def_var(ncid,"FGMEM", nf90_real,dimids,varid_fgmem))
    call check_return_code(nf90_def_var(ncid,"FLFMEM",nf90_real,dimids,varid_flfmem))
    call check_return_code(nf90_def_var(ncid,"FRFMEM",nf90_real,dimids,varid_frfmem))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid_finfl,&
                                        river_prognostic_fields%river_inflow))
    call check_return_code(nf90_put_var(ncid,varid_fgmem,&
                                        river_prognostic_fields%base_flow_reservoirs))
    call check_return_code(nf90_put_var(ncid,varid_flfmem,&
                                        river_prognostic_fields%overland_flow_reservoirs))
    call check_return_code(nf90_put_var(ncid,varid_frfmem,&
                                        river_prognostic_fields%river_flow_reservoirs))
    call check_return_code(nf90_close(ncid))
end subroutine write_river_initial_values

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

function load_drainages_fields(drainages_filename,first_timestep,last_timestep,&
                               river_parameters) &
    result(drainages)
  character(len = max_name_length) :: drainages_filename
  type(riverparameters), intent(in) :: river_parameters
  real, allocatable, dimension(:,:,:) :: drainages
  real, allocatable, dimension(:,:) :: drainages_on_timeslice
  integer :: first_timestep, last_timestep
  integer :: nlat,nlon
  integer :: ncid,varid
  integer, dimension(3) :: start
  integer, dimension(3) :: count
  integer :: t
    nlat = 360
    nlon = 720
    call check_return_code(nf90_open(drainages_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,"drainages", varid))
    allocate(drainages(nlat,nlon,last_timestep-first_timestep))
    count = (/river_parameters%nlat,river_parameters%nlon,1/)
    start = (/1,1,1/)
    do t = first_timestep,last_timestep
      start(3) = t
      call check_return_code(nf90_put_var(ncid,varid,drainages_on_timeslice,start,count))
      drainages(:,:,t) = drainages_on_timeslice(:,:)
    end do
    call check_return_code(nf90_close(ncid))
end function load_drainages_fields

function load_runoff_fields(runoffs_filename,first_timestep,last_timestep, &
                            river_parameters) &
    result(runoffs)
  character(len = max_name_length) :: runoffs_filename
  real, allocatable, dimension(:,:,:) :: runoffs
  real, allocatable, dimension(:,:) :: runoffs_on_timeslice
  type(riverparameters), intent(in) :: river_parameters
  integer :: nlat,nlon
  integer :: ncid,varid
  integer :: first_timestep, last_timestep
  integer, dimension(3) :: start
  integer, dimension(3) :: count
  integer :: t
    nlat = 360
    nlon = 720
    call check_return_code(nf90_open(runoffs_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,"runoffs", varid))
    allocate(runoffs(nlat,nlon,last_timestep-first_timestep))
    count = (/river_parameters%nlat,river_parameters%nlon,1/)
    start = (/1,1,1/)
    do t = first_timestep,last_timestep
      start(3) = t
      call check_return_code(nf90_put_var(ncid,varid,runoffs_on_timeslice,start,count))
      runoffs(:,:,t) = runoffs_on_timeslice(:,:)
    end do
    call check_return_code(nf90_close(ncid))
end function load_runoff_fields

! subroutine write_river_flow_field(river_parameters,river_flow_field,timestep)
!   type(riverparameters), pointer,intent(in) :: river_parameters
!   real, allocatable, dimension(:,:),intent(in) :: river_flow_field
!   integer,intent(in) :: timestep
!   integer :: ncid
!   character(len = max_name_length) :: filename
!   character(len = 50) :: timestep_str
!   if(timestep == -1) then
!     filename = '/Users/thomasriddick/Documents/data/temp/transient_sim_1/river_model_results.nc'
!   else
!     filename = '/Users/thomasriddick/Documents/data/temp/transient_sim_1/river_model_results_'
!     write (timestep_str,*) timestep
!     filename = trim(filename) // timestep_str // '.nc'
!   end if
!   call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
!   ! write_field(river_parameters.grid,"river_flow",river_flow_field,filename)
! end subroutine write_river_flow_field

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

end module latlon_hd_model_io_mod
