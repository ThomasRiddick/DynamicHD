module latlon_lake_logger_mod

implicit none

integer,parameter,private :: dp = selected_real_kind(12)

integer, parameter :: print_interval = 1000
integer, parameter :: first_logging_timestep = 9950
integer, parameter :: maximum_output_per_timestep = 10000

type latlon_lake_logger
  integer :: timestep
  integer, dimension(print_interval,3) :: lake_process_log
  real(dp) :: lake_volumes(print_interval)
  integer :: log_position
  integer :: timestep_total_output
  logical :: logging_status
  contains
    procedure :: initialise_latlon_lake_logger
    procedure :: increment_timestep
    procedure :: log_process
    procedure :: print_log
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
    this%logging_status = .false.
    this%log_position = 0
    this%timestep_total_output = 0
end subroutine initialise_latlon_lake_logger

subroutine increment_timestep(this)
  class(latlon_lake_logger) :: this
    this%timestep = this%timestep+1
    if (this%logging_status) then
        write(*,'(A,I6)') "Running lake timestep: ",this%timestep
        call this%print_log()
        this%timestep_total_output = 0
        this%lake_process_log(:,:) = 0
        this%lake_volumes(:) = 0
        this%log_position = 0
    end if
    if (this%timestep > first_logging_timestep) then
      this%logging_status = .true.
      this%timestep_total_output = 0
    end if
end subroutine increment_timestep

subroutine log_process(this,lake_number,lake_center_lat,lake_center_lon, &
                       lake_volume)
  class(latlon_lake_logger) :: this
  integer :: lake_number
  integer :: lake_center_lat
  integer :: lake_center_lon
  real(dp) :: lake_volume
    if(this%logging_status) then
      this%log_position = this%log_position + 1
      this%timestep_total_output = this%timestep_total_output + 1
      this%lake_process_log(this%log_position,1) = lake_number
      this%lake_process_log(this%log_position,2) = lake_center_lat
      this%lake_process_log(this%log_position,3) = lake_center_lon
      this%lake_volumes(this%log_position) = lake_volume
      if (this%timestep_total_output >= maximum_output_per_timestep) then
        call this%print_log()
        write(*,*) "!!!Maximum printout length for a single timestep reached!!!"
        this%lake_process_log(:,:) = 0
        this%lake_volumes(:) = 0
        this%log_position = 0
        this%logging_status = .false.
      else if (this%log_position >= print_interval) then
        call this%print_log()
        this%lake_process_log(:,:) = 0
        this%lake_volumes(:) = 0
        this%log_position = 0
      end if
    end if
end subroutine

subroutine print_log(this)
  class(latlon_lake_logger) :: this
  integer :: i
    do i = 1,this%log_position
      write(*,'(3(X,I8),3X,ES14.7)') this%lake_process_log(i,1),this%lake_process_log(i,2),&
                 this%lake_process_log(i,3),this%lake_volumes(i)
    end do
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
                               lake_volume)
  integer :: lake_number
  integer :: lake_center_lat
  integer :: lake_center_lon
  real(dp) :: lake_volume
    call global_lake_logger%log_process(lake_number,lake_center_lat,&
                                        lake_center_lon,lake_volume)
end subroutine log_process_wrapper

end module latlon_lake_logger_mod
