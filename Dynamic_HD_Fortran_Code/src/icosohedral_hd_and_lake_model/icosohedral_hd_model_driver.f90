program icosohedral_hd_model_driver

  use icosohedral_hd_model_interface_mod
  use parameters_mod
  implicit none

  real   ,dimension(:,:), allocatable :: runoffs
  real   ,dimension(:,:), allocatable :: drainages
  real   ,dimension(:,:), allocatable :: evaporations

  integer :: timesteps
  integer :: middle_timestep
  integer :: ncells
  logical :: using_lakes
  integer :: num_args

  character(len = max_name_length) :: river_params_filename
  character(len = max_name_length) :: hd_start_filename
  character(len = max_name_length) :: lake_model_ctl_filename
  character(len = max_name_length) :: timesteps_char
  character(len = max_name_length) :: ncells_char
  character(len = max_name_length) :: working_directory


    num_args = command_argument_count()
    if (num_args /= 5 .and. num_args /= 6) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=river_params_filename)
    call get_command_argument(2,value=hd_start_filename)
    call get_command_argument(3,value=timesteps_char)
    call get_command_argument(4,value=ncells_char)
    call get_command_argument(5,value=working_directory)
    if (num_args == 6) then
      using_lakes = .true.
      call get_command_argument(6,value=lake_model_ctl_filename)
    else
      using_lakes = .false.
      lake_model_ctl_filename = ""
    end if
    read(timesteps_char,*) timesteps
    read(ncells_char,*) ncells
    call init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                       lake_model_ctl_filename,86400.0,86400.0)
    allocate(runoffs(ncells,timesteps))
    allocate(drainages(ncells,timesteps))
    allocate(evaporations(ncells,timesteps))
    middle_timestep = timesteps/2
    runoffs(:,:) = 100.0*0.0000000227*86400.0*2.6*10000000000.0
    drainages(:,:) = 100.0*0.0000000227*86400.0*2.6*10000000000.0
    evaporations(:,1:middle_timestep) = 0.0
    evaporations(:,middle_timestep+1:timesteps) = 100000.0*0.0000000227*86400.0*2.6*10000000000.0
    call run_hd_model(timesteps,drainages,runoffs,evaporations,working_directory)
    call clean_hd_model()
    if (using_lakes) call clean_lake_model()
    deallocate(runoffs)
    deallocate(drainages)

end program icosohedral_hd_model_driver
