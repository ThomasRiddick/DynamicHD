module latlon_hd_model_io_mod

use netcdf
use latlon_hd_model_mod
use check_return_code_netcdf_mod
use parameters_mod
implicit none

contains

function read_river_parameters(river_params_filename) &
    result(river_parameters)
  character(len = max_name_length) :: river_params_filename
  type(riverparameters), pointer :: river_parameters
  real(dp), pointer, dimension(:,:) :: rdirs
  integer, pointer, dimension(:,:) :: river_reservoir_nums
  integer, pointer, dimension(:,:) :: overland_reservoir_nums
  integer, pointer, dimension(:,:) :: base_reservoir_nums
  real(dp), pointer, dimension(:,:) :: river_retention_coefficients
  real(dp), pointer, dimension(:,:) :: overland_retention_coefficients
  real(dp), pointer, dimension(:,:) :: base_retention_coefficients
  integer, pointer, dimension(:,:) :: landsea_mask_int
  logical, pointer, dimension(:,:) :: landsea_mask
  real(dp), allocatable, dimension(:,:) :: temp_real_array
  integer, allocatable, dimension(:,:) :: temp_integer_array
  integer, dimension(2) :: dimids
  integer :: ncid,varid
  integer :: nlat,nlon

    write(*,*) "Loading river parameters from: " // trim(river_params_filename)

    call check_return_code(nf90_open(river_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,'FDIR',varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=nlon))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(2),len=nlat))
    allocate(temp_real_array(nlon,nlat))
    allocate(temp_integer_array(nlon,nlat))
    allocate(rdirs(nlat,nlon))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    rdirs = transpose(temp_real_array)

    allocate(river_reservoir_nums(nlat,nlon))
    call check_return_code(nf90_inq_varid(ncid,'ARF_N',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    river_reservoir_nums = transpose(temp_integer_array)

    allocate(overland_reservoir_nums(nlat,nlon))
    call check_return_code(nf90_inq_varid(ncid,'ALF_N',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_integer_array))
    overland_reservoir_nums = transpose(temp_integer_array)

    allocate(base_reservoir_nums(nlat,nlon))
    base_reservoir_nums = 1

    allocate(river_retention_coefficients(nlat,nlon))
    call check_return_code(nf90_inq_varid(ncid,'ARF_K',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    river_retention_coefficients = transpose(temp_real_array)

    allocate(overland_retention_coefficients(nlat,nlon))
    call check_return_code(nf90_inq_varid(ncid,'ALF_K',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    overland_retention_coefficients = transpose(temp_real_array)

    allocate(base_retention_coefficients(nlat,nlon))
    call check_return_code(nf90_inq_varid(ncid,'AGF_K',varid))
    call check_return_code(nf90_get_var(ncid, varid,temp_real_array))
    base_retention_coefficients = transpose(temp_real_array)

    allocate(landsea_mask_int(nlat,nlon))
    call check_return_code(nf90_inq_varid(ncid,'FLAG',varid))
    call check_return_code(nf90_get_var(ncid,varid,temp_integer_array))
    landsea_mask_int = transpose(temp_integer_array)

    deallocate(temp_real_array)
    deallocate(temp_integer_array)

    allocate(landsea_mask(nlat,nlon))
    where (landsea_mask_int == 0)
      landsea_mask = .true.
    elsewhere
      landsea_mask = .false.
    end where
    deallocate(landsea_mask_int)

    call check_return_code(nf90_close(ncid))

    river_parameters => riverparameters(rdirs,river_reservoir_nums, &
                                        overland_reservoir_nums, &
                                        base_reservoir_nums, &
                                        river_retention_coefficients, &
                                        overland_retention_coefficients, &
                                        base_retention_coefficients, &
                                        landsea_mask)
end function read_river_parameters

function load_river_initial_values(hd_start_filename) &
    result(river_prognostic_fields)
  character(len = max_name_length) :: hd_start_filename
  type(riverprognosticfields), pointer :: river_prognostic_fields
  real(dp), pointer,     dimension(:,:) :: river_inflow
  real(dp), pointer, dimension(:,:) :: river_inflow_temp
  real(dp), pointer,     dimension(:,:,:) :: base_flow_reservoirs
  real(dp), pointer, dimension(:,:) :: base_flow_reservoirs_temp
  real(dp), pointer,    dimension(:,:,:) :: overland_flow_reservoirs
  real(dp), pointer, dimension(:,:) :: overland_flow_reservoirs_temp
  real(dp), pointer,     dimension(:,:,:) :: river_flow_reservoirs
  real(dp), pointer, dimension(:,:) :: river_flow_reservoirs_temp1
  real(dp), pointer, dimension(:,:) :: river_flow_reservoirs_temp2
  real(dp), pointer, dimension(:,:) :: river_flow_reservoirs_temp3
  real(dp), pointer, dimension(:,:) :: river_flow_reservoirs_temp4
  real(dp), pointer, dimension(:,:) :: river_flow_reservoirs_temp5
  integer, dimension(2) :: dimids
  integer :: nlat,nlon
  integer :: ncid,varid

    write(*,*) "Loading hd initial values from: " // trim(hd_start_filename)

    call check_return_code(nf90_open(hd_start_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,'FINFL',varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=nlon))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(2),len=nlat))
    allocate(river_inflow_temp(nlon,nlat))
    call check_return_code(nf90_get_var(ncid, varid,river_inflow_temp))

    allocate(base_flow_reservoirs_temp(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FGMEM',varid))
    call check_return_code(nf90_get_var(ncid, varid,base_flow_reservoirs_temp))

    allocate(overland_flow_reservoirs_temp(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FLFMEM',varid))
    call check_return_code(nf90_get_var(ncid, varid,overland_flow_reservoirs_temp))

    allocate(river_flow_reservoirs_temp1(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FRFMEM1',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp1))

    allocate(river_flow_reservoirs_temp2(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FRFMEM2',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp2))

    allocate(river_flow_reservoirs_temp3(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FRFMEM3',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp3))

    allocate(river_flow_reservoirs_temp4(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FRFMEM4',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp4))

    allocate(river_flow_reservoirs_temp5(nlon,nlat))
    call check_return_code(nf90_inq_varid(ncid,'FRFMEM5',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_flow_reservoirs_temp5))

    call check_return_code(nf90_close(ncid))

    allocate(river_inflow(nlat,nlon))
    allocate(base_flow_reservoirs(nlat,nlon,1))
    allocate(overland_flow_reservoirs(nlat,nlon,1))
    allocate(river_flow_reservoirs(nlat,nlon,5))
    river_inflow = transpose(river_inflow_temp)
    base_flow_reservoirs(:,:,1) = transpose(base_flow_reservoirs_temp(:,:))
    overland_flow_reservoirs(:,:,1) = transpose(overland_flow_reservoirs_temp(:,:))
    river_flow_reservoirs(:,:,1) = transpose(river_flow_reservoirs_temp1(:,:))
    river_flow_reservoirs(:,:,2) = transpose(river_flow_reservoirs_temp2(:,:))
    river_flow_reservoirs(:,:,3) = transpose(river_flow_reservoirs_temp3(:,:))
    river_flow_reservoirs(:,:,4) = transpose(river_flow_reservoirs_temp4(:,:))
    river_flow_reservoirs(:,:,5) = transpose(river_flow_reservoirs_temp5(:,:))
    river_prognostic_fields => riverprognosticfields(river_inflow, &
                                                     base_flow_reservoirs, &
                                                     overland_flow_reservoirs, &
                                                     river_flow_reservoirs)
    deallocate(river_inflow_temp)
    deallocate(base_flow_reservoirs_temp)
    deallocate(overland_flow_reservoirs_temp)
    deallocate(river_flow_reservoirs_temp1)
    deallocate(river_flow_reservoirs_temp2)
    deallocate(river_flow_reservoirs_temp3)
    deallocate(river_flow_reservoirs_temp4)
    deallocate(river_flow_reservoirs_temp5)
end function load_river_initial_values

subroutine write_river_initial_values(hd_start_filename,river_parameters, &
                                      river_prognostic_fields)
  character(len = max_name_length) :: hd_start_filename
  type(riverparameters), pointer :: river_parameters
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
  real(dp), pointer, dimension(:,:,:) :: drainages
  real(dp), pointer, dimension(:,:) :: drainages_on_timeslice
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
  real(dp), pointer, dimension(:,:,:) :: runoffs
  real(dp), pointer, dimension(:,:) :: runoffs_on_timeslice
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

subroutine write_river_flow_field(working_directory,river_parameters,&
                                  river_flow_field,timestep)
  character(len = *), intent(in) :: working_directory
  type(riverparameters), pointer,intent(in) :: river_parameters
  real(dp), pointer, dimension(:,:),intent(in) :: river_flow_field
  integer,intent(in) :: timestep
  integer :: ncid,varid
  integer :: lat_dimid,lon_dimid
  ! integer :: lat_varid,lon_varid
  integer, dimension(2) :: dimids
  character(len = max_name_length) :: filename
  character(len = 50) :: timestep_str
    filename = trim(working_directory) // '/river_model_results_'
    write (timestep_str,'(I0.3)') timestep
    filename = trim(filename) // trim(timestep_str) // '.nc'
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"lat",river_parameters%nlat,lat_dimid))
    call check_return_code(nf90_def_dim(ncid,"lon",river_parameters%nlon,lon_dimid))
    ! call check_return_code(nf90_def_var(ncid,"lat",nf90_double,lat_dimid,lat_varid))
    ! call check_return_code(nf90_def_var(ncid,"lon",nf90_double,lon_dimid,lon_varid))
    dimids = (/lat_dimid,lon_dimid/)
    call check_return_code(nf90_def_var(ncid,"hydro_discharge",nf90_real,dimids,varid))
    ! call check_return_code(nf90_put_var(ncid,lat_varid,))
    ! call check_return_code(nf90_put_var(ncid,lon_varid,))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,&
                                        river_flow_field))
    call check_return_code(nf90_close(ncid))
end subroutine write_river_flow_field

end module latlon_hd_model_io_mod
