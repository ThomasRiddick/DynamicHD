program latlon_hd_model_driver

  use latlon_hd_model_interface_mod
  use parameters_mod
  implicit none

  real(dp)   ,dimension(:,:,:), allocatable :: runoffs
  real(dp)   ,dimension(:,:,:), allocatable :: drainages
  real(dp)   ,dimension(:,:,:), allocatable :: evaporations

  integer :: timesteps
  integer :: middle_timestep ! the timestep half way through
  logical :: using_lakes
  integer :: num_args

  character(len = max_name_length) :: river_params_filename
  character(len = max_name_length) :: hd_start_filename
  character(len = max_name_length) :: lake_model_ctl_filename
  character(len = max_name_length) :: working_directory !Path to the working directory to use
  character(len = max_name_length) :: timesteps_char


    num_args = command_argument_count()
    if (num_args /= 4 .and. num_args /= 5) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=river_params_filename)
    call get_command_argument(2,value=hd_start_filename)
    call get_command_argument(3,value=timesteps_char)
    call get_command_argument(4,value=working_directory)
    if (num_args == 5) then
      using_lakes = .true.
      call get_command_argument(5,value=lake_model_ctl_filename)
    else
      using_lakes = .false.
      lake_model_ctl_filename = ""
    end if
    read(timesteps_char,*) timesteps
    call init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                       lake_model_ctl_filename)
    middle_timestep = timesteps/2
    allocate(runoffs(360,720,timesteps))
    allocate(drainages(360,720,timesteps))
    allocate(evaporations(48,96,timesteps))
    runoffs = 100.0_dp*0.0000000227_dp*2.6*10000000000.0_dp*20480.0_dp/259200.0_dp
    drainages = 100.0_dp*0.0000000227_dp*2.6*10000000000.0_dp*20480.0_dp/259200.0_dp
    evaporations(:,:,1:middle_timestep) = 0.0
    evaporations(:,:,middle_timestep+1:timesteps) = 1.0_dp
    call run_hd_model(timesteps,drainages,runoffs,evaporations,.true.,working_directory)
    call clean_hd_model()
    if (using_lakes) call clean_lake_model()
    deallocate(runoffs)
    deallocate(drainages)
    deallocate(evaporations)

end program latlon_hd_model_driver
