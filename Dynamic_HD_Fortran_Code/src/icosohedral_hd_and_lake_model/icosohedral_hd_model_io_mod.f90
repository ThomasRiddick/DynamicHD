module icosohedral_hd_model_io_mod

use netcdf
use icosohedral_hd_model_mod
use check_return_code_netcdf_mod
use grid_information_mod
implicit none

contains

function read_grid_information(river_params_filename) &
    result(grid_information)
  character(len = max_name_length) :: river_params_filename
  type(gridinformation) :: grid_information
  integer :: ncells
  real,pointer,dimension(:) :: clat
  real,pointer,dimension(:) :: clon
  real,pointer,dimension(:,:) :: clat_bounds
  real,pointer,dimension(:,:) :: clon_bounds
  integer :: ncid
  integer, dimension(2) :: dimids
  integer :: varid
  integer :: varid_clat,varid_clon
  integer :: varid_clat_bnds,varid_clon_bnds

    write(*,*) "Loading grid parameters from: " // trim(river_params_filename)

    call check_return_code(nf90_open(river_params_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,'FDIR',varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=ncells))
    call check_return_code(nf90_inq_varid(ncid,'clat',varid_clat))
    call check_return_code(nf90_inq_varid(ncid,'clon',varid_clon))
    call check_return_code(nf90_inq_varid(ncid,'clat_bnds',varid_clat_bnds))
    call check_return_code(nf90_inq_varid(ncid,'clon_bnds',varid_clon_bnds))
    allocate(clat(ncells))
    allocate(clon(ncells))
    allocate(clat_bounds(3,ncells))
    allocate(clon_bounds(3,ncells))
    call check_return_code(nf90_get_var(ncid, varid_clat,clat))
    call check_return_code(nf90_get_var(ncid, varid_clon,clon))
    call check_return_code(nf90_get_var(ncid, varid_clat_bnds,clat_bounds))
    call check_return_code(nf90_get_var(ncid, varid_clon_bnds,clon_bounds))
    grid_information = gridinformation(ncells,clat,clon,clat_bounds,clon_bounds)
end function read_grid_information

function read_river_parameters(river_params_filename,step_length,day_length) &
    result(river_parameters)
  character(len = max_name_length) :: river_params_filename
  real :: step_length
  real :: day_length
  type(riverparameters), pointer :: river_parameters
  integer, pointer, dimension(:) :: next_cell_index
  real, pointer, dimension(:)    :: temporary_real_array
  integer, pointer, dimension(:) :: river_reservoir_nums
  integer, pointer, dimension(:) :: overland_reservoir_nums
  integer, pointer, dimension(:) :: base_reservoir_nums
  real, pointer, dimension(:) :: river_retention_coefficients
  real, pointer, dimension(:) :: overland_retention_coefficients
  real, pointer, dimension(:) :: base_retention_coefficients
  integer, pointer, dimension(:) :: landsea_mask_int
  logical, pointer, dimension(:) :: landsea_mask
  integer, dimension(1) :: dimids
  integer :: ncid,varid
  integer :: ncells

    write(*,*) "Loading river parameters from: " // trim(river_params_filename)

    call check_return_code(nf90_open(river_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,'FDIR',varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=ncells))
    allocate(temporary_real_array(ncells))
    allocate(next_cell_index(ncells))
    call check_return_code(nf90_get_var(ncid, varid,temporary_real_array))
    next_cell_index = int(temporary_real_array)
    deallocate(temporary_real_array)

    allocate(river_reservoir_nums(ncells))
    call check_return_code(nf90_inq_varid(ncid,'ARF_N',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_reservoir_nums))

    allocate(overland_reservoir_nums(ncells))
    call check_return_code(nf90_inq_varid(ncid,'ALF_N',varid))
    call check_return_code(nf90_get_var(ncid, varid,overland_reservoir_nums))

    allocate(base_reservoir_nums(ncells))
    base_reservoir_nums(:) = 1

    allocate(river_retention_coefficients(ncells))
    call check_return_code(nf90_inq_varid(ncid,'ARF_K',varid))
    call check_return_code(nf90_get_var(ncid, varid,river_retention_coefficients))

    allocate(overland_retention_coefficients(ncells))
    call check_return_code(nf90_inq_varid(ncid,'ALF_K',varid))
    call check_return_code(nf90_get_var(ncid, varid,overland_retention_coefficients))

    allocate(base_retention_coefficients(ncells))
    call check_return_code(nf90_inq_varid(ncid,'AGF_K',varid))
    call check_return_code(nf90_get_var(ncid, varid,base_retention_coefficients))

    allocate(landsea_mask_int(ncells))
    call check_return_code(nf90_inq_varid(ncid,'MASK',varid))
    call check_return_code(nf90_get_var(ncid,varid,landsea_mask_int))

    allocate(landsea_mask(ncells))
    where (landsea_mask_int == 0)
      landsea_mask = .true.
    elsewhere
      landsea_mask = .false.
    end where
    deallocate(landsea_mask_int)

    call check_return_code(nf90_close(ncid))

    river_parameters => riverparameters(next_cell_index,river_reservoir_nums, &
                                        overland_reservoir_nums, &
                                        base_reservoir_nums, &
                                        river_retention_coefficients, &
                                        overland_retention_coefficients, &
                                        base_retention_coefficients, &
                                        landsea_mask,step_length,day_length)
end function read_river_parameters

function load_river_initial_values(hd_start_filename) &
    result(river_prognostic_fields)
  character(len = max_name_length) :: hd_start_filename
  type(riverprognosticfields), pointer :: river_prognostic_fields
  real, pointer,     dimension(:) :: river_inflow
  real, pointer,     dimension(:,:) :: base_flow_reservoirs
  real, pointer,    dimension(:,:) :: overland_flow_reservoirs
  real, pointer,     dimension(:,:) :: river_flow_reservoirs
  integer, dimension(2) :: dimids
  integer :: ncells
  integer :: ncid,varid

    write(*,*) "Loading hd initial values from: " // trim(hd_start_filename)

    call check_return_code(nf90_open(hd_start_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_varid(ncid,'FGMEM',varid))
    call check_return_code(nf90_inquire_variable(ncid,varid,dimids=dimids))
    call check_return_code(nf90_inquire_dimension(ncid,dimids(1),len=ncells))
    allocate(base_flow_reservoirs(ncells,1))
    call check_return_code(nf90_get_var(ncid, varid,base_flow_reservoirs))

    allocate(overland_flow_reservoirs(ncells,1))
    call check_return_code(nf90_inq_varid(ncid,'FLFMEM',varid))
    call check_return_code(nf90_get_var(ncid, varid,overland_flow_reservoirs))

    allocate(river_flow_reservoirs(ncells,5))
    call check_return_code(nf90_inq_varid(ncid,'FRFMEM',varid))
    call check_return_code(nf90_get_var(ncid,varid,river_flow_reservoirs))

    allocate(river_inflow(ncells))
    river_inflow(:) = 0.0

    call check_return_code(nf90_close(ncid))

    river_prognostic_fields => riverprognosticfields(river_inflow, &
                                                     base_flow_reservoirs, &
                                                     overland_flow_reservoirs, &
                                                     river_flow_reservoirs)
end function load_river_initial_values

subroutine write_river_initial_values(hd_start_filename,river_parameters, &
                                      river_prognostic_fields)
  character(len = max_name_length) :: hd_start_filename
  type(lakeparameters), pointer :: river_parameters
  type(riverprognosticfields), pointer :: river_prognostic_fields
  integer :: ncid,varid_fgmem
  integer :: varid_flfmem,varid_frfmem,dimid
  integer, dimension(1) :: dimids
    call check_return_code(nf90_create(hd_start_filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"ncells",river_parameters%ncells,dimid))
    dimids = (/dimid/)
    call check_return_code(nf90_def_var(ncid,"FGMEM", nf90_real,dimids,varid_fgmem))
    call check_return_code(nf90_def_var(ncid,"FLFMEM",nf90_real,dimids,varid_flfmem))
    call check_return_code(nf90_def_var(ncid,"FRFMEM",nf90_real,dimids,varid_frfmem))
    call check_return_code(nf90_enddef(ncid))
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
  real, pointer, dimension(:,:) :: drainages
  real, pointer, dimension(:) :: drainages_on_timeslice
  integer :: first_timestep, last_timestep
  integer :: ncells
  integer :: ncid,varid
  integer, dimension(2) :: start
  integer, dimension(2) :: count
  integer :: t
    ncells = 20480
    call check_return_code(nf90_open(drainages_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,"drainages", varid))
    allocate(drainages(ncells,last_timestep-first_timestep))
    count = (/river_parameters%ncells,1/)
    start = (/1,1/)
    do t = first_timestep,last_timestep
      start(2) = t
      call check_return_code(nf90_put_var(ncid,varid,drainages_on_timeslice,start,count))
      drainages(:,t) = drainages_on_timeslice(:)
    end do
    call check_return_code(nf90_close(ncid))
end function load_drainages_fields

function load_runoff_fields(runoffs_filename,first_timestep,last_timestep, &
                            river_parameters) &
    result(runoffs)
  character(len = max_name_length) :: runoffs_filename
  real, pointer, dimension(:,:) :: runoffs
  real, pointer, dimension(:) :: runoffs_on_timeslice
  type(riverparameters), intent(in) :: river_parameters
  integer :: ncells
  integer :: ncid,varid
  integer :: first_timestep, last_timestep
  integer, dimension(2) :: start
  integer, dimension(2) :: count
  integer :: t
    ncells = 20480
    call check_return_code(nf90_open(runoffs_filename,nf90_nowrite,ncid))
    call check_return_code(nf90_inq_varid(ncid,"runoffs", varid))
    allocate(runoffs(ncells,last_timestep-first_timestep))
    count = (/river_parameters%ncells,1/)
    start = (/1,1/)
    do t = first_timestep,last_timestep
      start(2) = t
      call check_return_code(nf90_put_var(ncid,varid,runoffs_on_timeslice,start,count))
      runoffs(:,t) = runoffs_on_timeslice(:)
    end do
    call check_return_code(nf90_close(ncid))
end function load_runoff_fields

subroutine write_river_flow_field(working_directory,river_parameters,&
                                  river_flow_field,timestep, &
                                  grid_information)
  character(len = *), intent(in) :: working_directory
  type(riverparameters), pointer,intent(in) :: river_parameters
  real, pointer, dimension(:),intent(in) :: river_flow_field
  integer,intent(in) :: timestep
  type(gridinformation), intent(in) :: grid_information
  integer :: ncid,varid
  integer :: varid_clat,varid_clon
  integer :: varid_clat_bnds,varid_clon_bnds
  integer :: dimid
  integer :: dimid_vert
  integer, dimension(1) :: dimids
  integer, dimension(2) :: dimids_bnds
  character(len = max_name_length*2) :: filename
  character(len = 50) :: timestep_str
    filename = trim(working_directory) // '/river_model_results_'
    write (timestep_str,'(I0.3)') timestep
    filename = trim(filename) // trim(timestep_str) // '.nc'
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"ncells",river_parameters%ncells,dimid))
    call check_return_code(nf90_def_dim(ncid,"vertices",3,dimid_vert))
    dimids = (/dimid/)
    dimids_bnds = (/dimid_vert,dimid/)
    call check_return_code(nf90_def_var(ncid,"hydro_discharge",nf90_double,dimids,varid))
    call check_return_code(nf90_def_var(ncid,"clat",nf90_double,dimids,varid_clat))
    call check_return_code(nf90_def_var(ncid,"clon",nf90_double,dimids,varid_clon))
    call check_return_code(nf90_def_var(ncid,"clat_bnds",nf90_double,dimids_bnds,varid_clat_bnds))
    call check_return_code(nf90_def_var(ncid,"clon_bnds",nf90_double,dimids_bnds,varid_clon_bnds))
    call check_return_code(nf90_put_att(ncid,varid_clat,"standard_name","latitude"))
    call check_return_code(nf90_put_att(ncid,varid_clat,"long_name","center latitude"))
    call check_return_code(nf90_put_att(ncid,varid_clat,"units","radian"))
    call check_return_code(nf90_put_att(ncid,varid_clat,"bounds","clat_bnds"))
    call check_return_code(nf90_put_att(ncid,varid_clon,"standard_name","longitude"))
    call check_return_code(nf90_put_att(ncid,varid_clon,"long_name","center longitude"))
    call check_return_code(nf90_put_att(ncid,varid_clon,"units","radian"))
    call check_return_code(nf90_put_att(ncid,varid_clon,"bounds","clon_bnds"))
    call check_return_code(nf90_put_att(ncid,varid,"standard_name","river discharge"))
    call check_return_code(nf90_put_att(ncid,varid,"units","m^3/s"))
    call check_return_code(nf90_put_att(ncid,varid,"grid_type","unstructured"))
    call check_return_code(nf90_put_att(ncid,varid,"coordinates","clat clon"))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,&
                                        river_flow_field))
    call check_return_code(nf90_put_var(ncid,varid_clat,&
                                        grid_information%clat))
    call check_return_code(nf90_put_var(ncid,varid_clon,&
                                        grid_information%clon))
    call check_return_code(nf90_put_var(ncid,varid_clat_bnds,&
                                        grid_information%clat_bounds))
    call check_return_code(nf90_put_var(ncid,varid_clon_bnds,&
                                        grid_information%clon_bounds))
    call check_return_code(nf90_close(ncid))
end subroutine write_river_flow_field

end module icosohedral_hd_model_io_mod
