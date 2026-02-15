program latlon_hd_model_driver

  use latlon_hd_model_interface_mod
  use lake_model_input, only: load_cell_areas_on_surface_model_grid
  use parameters_mod
  implicit none

  real(dp)   ,dimension(:,:,:), allocatable :: runoffs
  real(dp)   ,dimension(:,:,:), allocatable :: drainages
  real(dp)   ,dimension(:,:,:), allocatable :: evaporations
  real(dp)   ,dimension(:,:),   pointer :: cell_areas_on_surface_model_grid

  integer :: timesteps
  integer :: middle_timestep ! the timestep half way through
  logical :: using_lakes
  logical :: using_jsb_lake_interface
  integer :: num_args

  character(len = max_name_length) :: river_params_filename
  character(len = max_name_length) :: hd_start_filename
  character(len = max_name_length) :: lake_model_ctl_filename
  character(len = max_name_length) :: surface_cell_areas_filename
  character(len = max_name_length) :: working_directory !Path to the working directory to use
  character(len = max_name_length) :: timesteps_char
  character(len = max_name_length) :: using_jsb_lake_interface_char


    num_args = command_argument_count()
    if (num_args /= 4 .and. num_args /= 6 .and. num_args /= 7) then
      write(*,*) "Wrong number of command line arguments given"
      stop
    end if
    call get_command_argument(1,value=river_params_filename)
    call get_command_argument(2,value=hd_start_filename)
    call get_command_argument(3,value=timesteps_char)
    call get_command_argument(4,value=working_directory)
    if (num_args == 6 .or. num_args == 7) then
      using_lakes = .true.
      call get_command_argument(5,value=lake_model_ctl_filename)
      call get_command_argument(6,value=surface_cell_areas_filename)
      call get_command_argument(7,value=using_jsb_lake_interface_char)
      cell_areas_on_surface_model_grid => &
        load_cell_areas_on_surface_model_grid(surface_cell_areas_filename,96,48)
        !load_cell_areas_on_surface_model_grid(surface_cell_areas_filename,48,96)
      if (num_args == 7) then
        read(using_jsb_lake_interface_char,*) using_jsb_lake_interface
        if (using_jsb_lake_interface) then
          write(*,*) "Using jsbach interface"
        end if
      else
        using_jsb_lake_interface = .false.
      end if
    else
      using_lakes = .false.
      using_jsb_lake_interface = .false.
      lake_model_ctl_filename = ""
      cell_areas_on_surface_model_grid => null()
    end if
    read(timesteps_char,*) timesteps
    call init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                       using_jsb_lake_interface,&
                       lake_model_ctl_filename,cell_areas_on_surface_model_grid,&
                       360,720,48,96)
    middle_timestep = timesteps/2
    allocate(runoffs(360,720,timesteps))
    allocate(drainages(360,720,timesteps))
    allocate(evaporations(48,96,timesteps))
    runoffs = 0.0_dp
    !runoffs = 100.0_dp*0.0000000227_dp*2.6*10000000000.0_dp*20480.0_dp/259200.0_dp
    drainages = 0.0_dp
    !drainages = 100.0_dp*0.0000000227_dp*2.6*10000000000.0_dp*20480.0_dp/259200.0_dp
    evaporations(:,:,1:middle_timestep) = 0.0_dp
    evaporations(:,:,middle_timestep+1:timesteps) = 0.0_dp
    !evaporations(:,:,middle_timestep+1:timesteps) = 100.0_dp*0.0000000227_dp*2.6*10000000000.0_dp*20480.0_dp/259200.0_dp*100000.0_dp
    call run_hd_model(timesteps,drainages,runoffs,evaporations,.true.,working_directory)
    call clean_hd_model()
    if (using_lakes) call clean_lake_model(.true.)
    deallocate(runoffs)
    deallocate(drainages)
    deallocate(evaporations)
    deallocate(cell_areas_on_surface_model_grid)

end program latlon_hd_model_driver
