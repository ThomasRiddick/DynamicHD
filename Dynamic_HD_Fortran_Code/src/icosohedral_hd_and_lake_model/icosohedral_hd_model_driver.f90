program icosohedral_hd_model_driver

  use icosohedral_hd_model_interface_mod
  use parameters_mod
  implicit none

  real   ,dimension(:,:), allocatable :: runoffs
  real   ,dimension(:,:), allocatable :: drainages

  integer :: timesteps
  integer :: ncells
  logical :: using_lakes
  integer :: num_args

  character(len = max_name_length) :: river_params_filename
  character(len = max_name_length) :: hd_start_filename
  character(len = max_name_length) :: lake_model_ctl_filename
  character(len = max_name_length) :: timesteps_char
  character(len = max_name_length) :: ncells_char


    num_args = command_argument_count()
    if (num_args /= 4 .and. num_args /= 5) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=river_params_filename)
    call get_command_argument(2,value=hd_start_filename)
    call get_command_argument(3,value=timesteps_char)
    call get_command_argument(3,value=ncells_char)
    if (num_args == 5) then
      using_lakes = .true.
      call get_command_argument(5,value=lake_model_ctl_filename)
    else
      using_lakes = .false.
      lake_model_ctl_filename = ""
    end if
    read(timesteps_char,*) timesteps
    read(ncells_char,*) ncells
    call init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                       lake_model_ctl_filename)
    allocate(runoffs(ncells,timesteps))
    allocate(drainages(ncells,timesteps))
    runoffs(:,:) = 1.0
    drainages(:,:) = 1.0
    call run_hd_model(timesteps,drainages,runoffs)
    call clean_hd_model()
    if (using_lakes) call clean_lake_model()
    deallocate(runoffs)
    deallocate(drainages)

end program icosohedral_hd_model_driver
