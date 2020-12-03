module icosohedral_lake_model_io_mod

!This module contains the detailed Input/Output routines for the
!lake model

use netcdf
use icosohedral_lake_model_mod
use grid_information_mod
use check_return_code_netcdf_mod
use parameters_mod
implicit none

character(len = max_name_length) :: lake_params_filename
character(len = max_name_length) :: lake_start_filename
real(dp) :: lake_retention_coefficient

contains

! Add an offset to a c-style array index excepting a list of possible
! exceptions
subroutine add_offset(array,offset,exceptions)
  integer, dimension(:), intent(inout) :: array
  integer, intent(in) :: offset
  integer, dimension(:), optional, intent(in) :: exceptions
  logical, dimension(:), allocatable :: exception_mask
  integer :: exception
  integer :: i
    allocate(exception_mask(size(array)))
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

! Load the lake namelist
subroutine config_lakes(lake_model_ctl_filename)
  include 'lake_model_ctl.inc'

  character(len = max_name_length) :: lake_model_ctl_filename
  integer :: unit_number

    write(*,*) "Reading namelist: " // trim(lake_model_ctl_filename)
    open(newunit=unit_number,file=lake_model_ctl_filename,status='old')
    read (unit=unit_number,nml=lake_model_ctl)
    close(unit_number)

end subroutine config_lakes

! Read the set of pre-calculated parameters for the lakes from the file set in the
! lake_params_filename module variable. Takes the instant_throughflow flag as an input
! and adds it to the lake parameters object created
function read_lake_parameters(instant_throughflow)&
                              result(lake_parameters)
  logical, intent(in) :: instant_throughflow
  type(lakeparameters), pointer :: lake_parameters
  logical, pointer, dimension(:) :: lake_centers
  integer, pointer, dimension(:) :: lake_centers_int
  real(dp), pointer, dimension(:) :: connection_volume_thresholds
  real(dp), pointer, dimension(:) :: flood_volume_thresholds
  logical, pointer, dimension(:) :: flood_local_redirect
  integer, pointer, dimension(:) :: flood_local_redirect_int
  logical, pointer, dimension(:) :: connect_local_redirect
  integer, pointer, dimension(:) :: connect_local_redirect_int
  logical, pointer, dimension(:) :: additional_flood_local_redirect
  integer, pointer, dimension(:) :: additional_flood_local_redirect_int
  logical, pointer, dimension(:) :: additional_connect_local_redirect
  integer, pointer, dimension(:) :: additional_connect_local_redirect_int
  integer, pointer, dimension(:) :: merge_points
  integer, pointer, dimension(:) :: flood_next_cell_index
  integer, pointer, dimension(:) :: connect_next_cell_index
  integer, pointer, dimension(:) :: flood_force_merge_index
  integer, pointer, dimension(:) :: connect_force_merge_index
  integer, pointer, dimension(:) :: flood_redirect_index
  integer, pointer, dimension(:) :: connect_redirect_index
  integer, pointer, dimension(:) :: additional_flood_redirect_index
  integer, pointer, dimension(:) :: additional_connect_redirect_index
  integer :: ncells
  integer :: ncells_coarse
  integer :: ncid
  integer :: varid,dimid

    write(*,*) "Loading lake parameters from file: " // trim(lake_params_filename)

    call check_return_code(nf90_open(lake_params_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'ncells',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells))
    ncells_coarse = ncells

    allocate(lake_centers_int(ncells))
    call check_return_code(nf90_inq_varid(ncid,'lake_centers',varid))
    call check_return_code(nf90_get_var(ncid, varid,lake_centers_int))

    allocate(connection_volume_thresholds(ncells))
    call check_return_code(nf90_inq_varid(ncid,'connection_volume_thresholds',varid))
    call check_return_code(nf90_get_var(ncid, varid,connection_volume_thresholds))

    allocate(flood_volume_thresholds(ncells))
    call check_return_code(nf90_inq_varid(ncid,'flood_volume_thresholds',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_volume_thresholds))

    allocate(flood_local_redirect_int(ncells))
    call check_return_code(nf90_inq_varid(ncid,'flood_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_local_redirect_int))

    allocate(connect_local_redirect_int(ncells))
    call check_return_code(nf90_inq_varid(ncid,'connect_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_local_redirect_int))

    allocate(additional_flood_local_redirect_int(ncells))
    call check_return_code(nf90_inq_varid(ncid,'additional_flood_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_flood_local_redirect_int))

    allocate(additional_connect_local_redirect_int(ncells))
    call check_return_code(nf90_inq_varid(ncid,'additional_connect_local_redirect',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_connect_local_redirect_int))

    allocate(flood_next_cell_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'flood_next_cell_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_next_cell_index))

    allocate(connect_next_cell_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'connect_next_cell_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_next_cell_index))

    allocate(flood_force_merge_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'flood_force_merge_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_force_merge_index))

    allocate(connect_force_merge_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'connect_force_merge_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_force_merge_index))

    allocate(flood_redirect_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'flood_redirect_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,flood_redirect_index))

    allocate(connect_redirect_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'connect_redirect_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,connect_redirect_index))

    allocate(additional_flood_redirect_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'additional_flood_redirect_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_flood_redirect_index))

    allocate(additional_connect_redirect_index(ncells))
    call check_return_code(nf90_inq_varid(ncid,'additional_connect_redirect_index',varid))
    call check_return_code(nf90_get_var(ncid, varid,additional_connect_redirect_index))

    allocate(merge_points(ncells))
    call check_return_code(nf90_inq_varid(ncid,'merge_points',varid))
    call check_return_code(nf90_get_var(ncid, varid,merge_points))

    call check_return_code(nf90_close(ncid))

    allocate(lake_centers(ncells))
    where (lake_centers_int == 0)
      lake_centers = .false.
    elsewhere
      lake_centers = .true.
    end where

    allocate(flood_local_redirect(ncells))
    where (flood_local_redirect_int == 0)
      flood_local_redirect = .false.
    elsewhere
      flood_local_redirect = .true.
    end where

    allocate(connect_local_redirect(ncells))
    where (connect_local_redirect_int == 0)
      connect_local_redirect = .false.
    elsewhere
      connect_local_redirect = .true.
    end where

    allocate(additional_flood_local_redirect(ncells))
    where (additional_flood_local_redirect_int == 0)
      additional_flood_local_redirect = .false.
    elsewhere
      additional_flood_local_redirect = .true.
    end where

    allocate(additional_connect_local_redirect(ncells))
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
                                      flood_next_cell_index, &
                                      connect_next_cell_index, &
                                      flood_force_merge_index, &
                                      connect_force_merge_index, &
                                      flood_redirect_index, &
                                      connect_redirect_index, &
                                      additional_flood_redirect_index, &
                                      additional_connect_redirect_index, &
                                      ncells, &
                                      ncells_coarse, &
                                      instant_throughflow, &
                                      lake_retention_coefficient)
end function read_lake_parameters

! Read the initial water to place into lakes and water lakes that used to exist that
! needs to be returned to the HD model
subroutine load_lake_initial_values(initial_water_to_lake_centers,&
                                    initial_spillover_to_rivers)
  real(dp), pointer, dimension(:), intent(inout) :: initial_water_to_lake_centers
  real(dp), pointer, dimension(:), intent(inout) :: initial_spillover_to_rivers
  integer :: ncells_coarse
  integer :: ncells
  integer :: ncid,varid,dimid

    write(*,*) "Loading lake initial values from file: " // trim(lake_start_filename)
    call check_return_code(nf90_open(lake_start_filename,nf90_nowrite,ncid))

    call check_return_code(nf90_inq_dimid(ncid,'cell',dimid))
    call check_return_code(nf90_inquire_dimension(ncid,dimid,len=ncells))
    ncells_coarse = ncells

    allocate(initial_water_to_lake_centers(ncells))
    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_lakes',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_water_to_lake_centers))

    allocate(initial_spillover_to_rivers(ncells_coarse))
    call check_return_code(nf90_inq_varid(ncid,'water_redistributed_to_rivers',varid))
    call check_return_code(nf90_get_var(ncid, varid,initial_spillover_to_rivers))

    call check_return_code(nf90_close(ncid))
end subroutine load_lake_initial_values

! Write the lake volumes in format that can be post processed to create a new lake
! start files
subroutine write_lake_volumes_field(lake_volumes_filename,&
                                    lake_parameters,lake_volumes)
  character(len = max_name_length) :: lake_volumes_filename
  real(dp), pointer, dimension(:), intent(inout) :: lake_volumes
  type(lakeparameters), pointer :: lake_parameters
  integer :: ncid,varid,dimid
  integer, dimension(1) :: dimids
    call check_return_code(nf90_create(lake_volumes_filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid, "ncells",lake_parameters%ncells,dimid))
    dimids = (/dimid/)
    call check_return_code(nf90_def_var(ncid, "lake_field", nf90_real, dimids,varid))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,lake_volumes))
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_volumes_field

! Write out the ID number of each point with a lake
subroutine write_lake_numbers_field(working_directory, &
                                    lake_parameters,lake_fields,&
                                    timestep,grid_information)
  character(len = *), intent(in) :: working_directory
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakefields), pointer, intent(in) :: lake_fields
  type(gridinformation), intent(in) :: grid_information
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
  integer :: ncid,varid,dimid,dimid_vert
  integer :: varid_clat,varid_clon
  integer :: varid_clat_bnds,varid_clon_bnds
  integer, dimension(1) :: dimids
  integer, dimension(2) :: dimids_bnds
    if(timestep == -1) then
      filename = trim(working_directory) // '/lake_model_results.nc'
    else
      filename = trim(working_directory) // '/lake_model_results_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"ncells",lake_parameters%ncells,dimid))
    call check_return_code(nf90_def_dim(ncid,"vertices",3,dimid_vert))
    dimids = (/dimid/)
    dimids_bnds = (/dimid_vert,dimid/)
    call check_return_code(nf90_def_var(ncid,"lake_number",nf90_int,dimids,varid))
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
    call check_return_code(nf90_put_att(ncid,varid,"standard_name","lake number"))
    call check_return_code(nf90_put_att(ncid,varid,"grid_type","unstructured"))
    call check_return_code(nf90_put_att(ncid,varid,"coordinates","clat clon"))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,&
                                        lake_fields%lake_numbers))
    call check_return_code(nf90_put_var(ncid,varid_clat,&
                                        grid_information%clat))
    call check_return_code(nf90_put_var(ncid,varid_clon,&
                                        grid_information%clon))
    call check_return_code(nf90_put_var(ncid,varid_clat_bnds,&
                                        grid_information%clat_bounds))
    call check_return_code(nf90_put_var(ncid,varid_clon_bnds,&
                                        grid_information%clon_bounds))
    call check_return_code(nf90_close(ncid))
end subroutine write_lake_numbers_field

!Write out the full volume of the lake (including all sub-basin) to each point
!the lake covers
subroutine write_diagnostic_lake_volumes(working_directory, &
                                         lake_parameters, &
                                         lake_prognostics, &
                                         lake_fields, &
                                         timestep,grid_information)
  character(len = *), intent(in) :: working_directory
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakeprognostics),pointer, intent(in) :: lake_prognostics
  type(lakefields), pointer, intent(in) :: lake_fields
  type(gridinformation) :: grid_information
  integer, intent(in) :: timestep
  character(len = 50) :: timestep_str
  character(len = max_name_length) :: filename
    integer :: ncid,varid,dimid,dimid_vert
  integer :: varid_clat,varid_clon
  integer :: varid_clat_bnds,varid_clon_bnds
  integer, dimension(1) :: dimids
  integer, dimension(2) :: dimids_bnds
  real(dp), dimension(:), pointer :: diagnostic_lake_volume
    diagnostic_lake_volume => calculate_diagnostic_lake_volumes(lake_parameters,&
                                                                lake_prognostics,&
                                                                lake_fields)
    if(timestep == -1) then
      filename = trim(working_directory) // '/diagnostic_lake_volume_results.nc'
    else
      filename = trim(working_directory) // '/diagnostic_lake_volume_results_'
      write (timestep_str,'(I0.3)') timestep
      filename = trim(filename) // trim(timestep_str) // '.nc'
    end if
    call check_return_code(nf90_create(filename,nf90_noclobber,ncid))
    call check_return_code(nf90_def_dim(ncid,"ncells",lake_parameters%ncells,dimid))
    call check_return_code(nf90_def_dim(ncid,"vertices",3,dimid_vert))
    dimids = (/dimid/)
    dimids_bnds = (/dimid_vert,dimid/)
    call check_return_code(nf90_def_var(ncid,"diagnostic_lake_volume",nf90_real,dimids,varid))
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
    call check_return_code(nf90_put_att(ncid,varid,"standard_name","lake number"))
    call check_return_code(nf90_put_att(ncid,varid,"grid_type","unstructured"))
    call check_return_code(nf90_put_att(ncid,varid,"coordinates","clat clon"))
    call check_return_code(nf90_enddef(ncid))
    call check_return_code(nf90_put_var(ncid,varid,diagnostic_lake_volume))
    call check_return_code(nf90_put_var(ncid,varid_clat,&
                                        grid_information%clat))
    call check_return_code(nf90_put_var(ncid,varid_clon,&
                                        grid_information%clon))
    call check_return_code(nf90_put_var(ncid,varid_clat_bnds,&
                                        grid_information%clat_bounds))
    call check_return_code(nf90_put_var(ncid,varid_clon_bnds,&
                                        grid_information%clon_bounds))
    call check_return_code(nf90_close(ncid))
    deallocate(diagnostic_lake_volume)
end subroutine write_diagnostic_lake_volumes

end module icosohedral_lake_model_io_mod
