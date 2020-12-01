program icosohedral_hd_model_driver

  !This driver runs a technical demonstration of the HD and lake model
  !using files input on the command line and setting a global constant
  !run-off and drainage and half way through adding a strong constant
  !evaporation

  use icosohedral_hd_model_interface_mod
  use parameters_mod
  implicit none

  real   ,dimension(:,:), allocatable :: runoffs !Runoff forcing
  real   ,dimension(:,:), allocatable :: drainages !Drainage forcing
  real   ,dimension(:,:), allocatable :: evaporations !Evaporation forcing

  integer :: timesteps !the number of time steps to run
  integer :: middle_timestep ! the timestep half way through
  integer :: ncells !The number of icosohedral cells in the grid being used
  logical :: using_lakes !Flag for using lakes
  integer :: num_args !The number of command line arguments

  character(len = max_name_length) :: river_params_filename ! hdpara filename
  character(len = max_name_length) :: hd_start_filename ! hdstart filename
  character(len = max_name_length) :: lake_model_ctl_filename !Control file for the lake
                                                              !model
  character(len = max_name_length) :: timesteps_char !The number of timesteps as a string
  character(len = max_name_length) :: ncells_char !The number of cells in the grid as a string
  character(len = max_name_length) :: working_directory !Path to the working directory to use

    ! Read the command line arguments
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

    !Initialise the hd model
    call init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                       lake_model_ctl_filename,86400.0,86400.0)
    ! Preparing some toy forcing for a technical demonstration
    ! For the first half of the run-time a constant level of runoff and drainage
    ! for the second half of run-time continue these but add in a very high
    ! level of evaporation. Hence lakes should grow up to the mid-point (and overflow)
    ! before it is reached and then evaporate after it and shrink back to nothing
    allocate(runoffs(ncells,timesteps))
    allocate(drainages(ncells,timesteps))
    allocate(evaporations(ncells,timesteps))
    middle_timestep = timesteps/2
    runoffs(:,:) = 100.0*0.0000000227*86400.0*2.6*10000000000.0
    drainages(:,:) = 100.0*0.0000000227*86400.0*2.6*10000000000.0
    evaporations(:,1:middle_timestep) = 0.0
    evaporations(:,middle_timestep+1:timesteps) = 100000.0*0.0000000227*86400.0*2.6*10000000000.0
    ! Run the hd model
    call run_hd_model(timesteps,drainages,runoffs,evaporations,working_directory)
    ! Free memory
    call clean_hd_model()
    if (using_lakes) call clean_lake_model()
    deallocate(runoffs)
    deallocate(drainages)

end program icosohedral_hd_model_driver
