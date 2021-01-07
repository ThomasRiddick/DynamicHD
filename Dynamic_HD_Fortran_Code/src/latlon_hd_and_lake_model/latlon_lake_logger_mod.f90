module latlon_lake_logger_mod

use netcdf
use check_return_code_netcdf_mod

implicit none

integer,parameter,private :: dp = selected_real_kind(12)

integer, parameter :: print_interval = 1000
integer, parameter :: first_logging_timestep = 9950
integer, parameter :: maximum_output_per_timestep = 10000
integer, parameter :: info_dump_timestep = 10955

type latlon_lake_logger
  integer :: timestep
  integer, dimension(print_interval,4) :: lake_process_log
  real(dp) :: lake_volumes(print_interval)
  character(len=30), dimension(print_interval) :: process_names
  integer :: log_position
  integer :: timestep_total_output
  logical :: logging_status
  contains
    procedure :: initialise_latlon_lake_logger
    procedure :: increment_timestep
    procedure :: log_process
    procedure :: print_log
    procedure :: create_info_dump
end type latlon_lake_logger

interface latlon_lake_logger
  procedure :: latlon_lake_logger_constructor
end interface latlon_lake_logger

type(latlon_lake_logger),pointer :: global_lake_logger


contains

subroutine initialise_latlon_lake_logger(this)
  class(latlon_lake_logger) :: this
    this%timestep = 0
    this%lake_process_log(:,:) = 0
    this%lake_volumes(:) = 0
    this%process_names(:) = ""
    this%logging_status = .false.
    this%log_position = 0
    this%timestep_total_output = 0
end subroutine initialise_latlon_lake_logger

subroutine increment_timestep(this)
  class(latlon_lake_logger) :: this
    this%timestep = this%timestep+1
    if (this%logging_status) then
        call this%print_log()
        write(*,'(A,I6)') "Running lake timestep: ",this%timestep
        this%timestep_total_output = 0
        this%lake_process_log(:,:) = 0
        this%lake_volumes(:) = 0
        this%process_names(:) = ""
        this%log_position = 0
    end if
    if (this%timestep > first_logging_timestep) then
      this%logging_status = .true.
      this%timestep_total_output = 0
    end if
end subroutine increment_timestep

subroutine create_info_dump(this,water_to_lakes,lake_volumes, &
                            nlat,nlon,nlat_coarse,nlon_coarse)
  class(latlon_lake_logger) :: this
  real(dp), dimension(:,:),pointer :: water_to_lakes
  real(dp), dimension(:,:),pointer :: lake_volumes
  integer ::  nlat,nlon,nlat_coarse,nlon_coarse
  character(len = *),parameter :: info_dump_filename = &
    "/Users/thomasriddick/Documents/data/temp/info_dump.nc"
  integer :: ncid,varid_volumes,varid_water_to_lakes
  integer :: lat_dimid,lon_dimid,lat_dimid_coarse,lon_dimid_coarse
  integer, dimension(2) :: dimids,dimids_coarse
    if(this%timestep == info_dump_timestep) then
        call check_return_code(nf90_create(info_dump_filename,nf90_noclobber,ncid))
        call check_return_code(nf90_def_dim(ncid, "lat",nlat,lat_dimid))
        call check_return_code(nf90_def_dim(ncid, "lon",nlon,lon_dimid))
        call check_return_code(nf90_def_dim(ncid, "lat_coarse",nlat_coarse,&
                                            lat_dimid_coarse))
        call check_return_code(nf90_def_dim(ncid, "lon_coarse",nlon_coarse,&
                                            lon_dimid_coarse))
        dimids = (/lat_dimid,lon_dimid/)
        dimids_coarse = (/lat_dimid_coarse,lon_dimid_coarse/)
        call check_return_code(nf90_def_var(ncid, "lake_volumes",&
                                            nf90_real, dimids,varid_volumes))
        call check_return_code(nf90_def_var(ncid, "water_to_lakes",&
                                            nf90_real, dimids_coarse,&
                                            varid_water_to_lakes))
        call check_return_code(nf90_enddef(ncid))
        call check_return_code(nf90_put_var(ncid,varid_volumes,lake_volumes))
        call check_return_code(nf90_put_var(ncid,varid_water_to_lakes,&
                                            water_to_lakes))
        call check_return_code(nf90_close(ncid))
    end if
    deallocate(lake_volumes)
end subroutine create_info_dump

subroutine log_process(this,lake_number,lake_center_lat,lake_center_lon, &
                       lake_type,lake_volume,process_name)
  class(latlon_lake_logger) :: this
  integer :: lake_number
  integer :: lake_center_lat
  integer :: lake_center_lon
  integer :: lake_type
  real(dp) :: lake_volume
  character(len=*) :: process_name
  character(len=30) :: process_name_fixed_length
    write(process_name_fixed_length,*) process_name
    if(this%logging_status) then
      this%log_position = this%log_position + 1
      this%timestep_total_output = this%timestep_total_output + 1
      this%lake_process_log(this%log_position,1) = lake_number
      this%lake_process_log(this%log_position,2) = lake_center_lat
      this%lake_process_log(this%log_position,3) = lake_center_lon
      this%lake_process_log(this%log_position,4) = lake_type
      this%lake_volumes(this%log_position) = lake_volume
      this%process_names(this%log_position) = process_name
      if (this%timestep_total_output >= maximum_output_per_timestep) then
        call this%print_log()
        write(*,*) "!!!Maximum printout length for a single timestep reached!!!"
        this%lake_process_log(:,:) = 0
        this%lake_volumes(:) = 0
        this%process_names(:) = ""
        this%log_position = 0
        this%logging_status = .false.
      else if (this%log_position >= print_interval) then
        call this%print_log()
        this%lake_process_log(:,:) = 0
        this%lake_volumes(:) = 0
        this%process_names(:) = ""
        this%log_position = 0
      end if
    end if
end subroutine

subroutine print_log(this)
  class(latlon_lake_logger) :: this
  integer :: i
    if (this%log_position > 0) then
      do i = 1,this%log_position
        write(*,'(4(X,I8),3X,ES14.7,3X,A)') this%lake_process_log(i,1),this%lake_process_log(i,2),&
                   this%lake_process_log(i,3),this%lake_process_log(i,4),&
                   this%lake_volumes(i),this%process_names(i)
      end do
    end if
end subroutine print_log

function latlon_lake_logger_constructor() result(constructor)
  type(latlon_lake_logger),pointer :: constructor
    allocate(constructor)
    call constructor%initialise_latlon_lake_logger()
end function latlon_lake_logger_constructor

subroutine setup_logger()
    global_lake_logger => latlon_lake_logger()
end subroutine setup_logger

subroutine increment_timestep_wrapper()
    call global_lake_logger%increment_timestep()
end subroutine increment_timestep_wrapper

subroutine log_process_wrapper(lake_number,lake_center_lat,lake_center_lon,&
                               lake_type,lake_volume,process_name)
  integer :: lake_number
  integer :: lake_center_lat
  integer :: lake_center_lon
  integer :: lake_type
  real(dp) :: lake_volume
  character(len=*) :: process_name
    call global_lake_logger%log_process(lake_number,lake_center_lat,&
                                        lake_center_lon,lake_type,&
                                        lake_volume,process_name)
end subroutine log_process_wrapper

subroutine create_info_dump_wrapper(water_to_lakes,lake_volumes, &
                                    nlat,nlon,nlat_coarse,nlon_coarse)
  real(dp), dimension(:,:),pointer :: water_to_lakes
  real(dp), dimension(:,:),pointer :: lake_volumes
  integer ::  nlat,nlon,nlat_coarse,nlon_coarse
    call global_lake_logger%create_info_dump(water_to_lakes,lake_volumes, &
                                             nlat,nlon,nlat_coarse,nlon_coarse)
end subroutine create_info_dump_wrapper

function is_logging()
  logical :: is_logging
    is_logging = global_lake_logger%logging_status
end function

end module latlon_lake_logger_mod
