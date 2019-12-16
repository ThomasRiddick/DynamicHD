module latlon_hd_model_io_mod

use netcdf
use latlon_hd_model_mod
use check_return_code_netcdf_mod
implicit none

contains

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

end module latlon_hd_model_io_mod
