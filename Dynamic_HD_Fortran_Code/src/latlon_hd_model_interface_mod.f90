module latlon_hd_model_interface_mod

  use latlon_hd_model_mod
  use latlon_hd_model_io_mod

  implicit none

  type(prognostics), target :: global_prognostics
  logical :: write_output

  contains

  subroutine init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                           lake_model_ctl_filename)
    logical, intent(in) :: using_lakes
    character(len = *) :: river_params_filename
    character(len = *) :: hd_start_filename
    character(len = *), optional :: lake_model_ctl_filename
    real, pointer, dimension(:,:) :: initial_spillover_to_rivers
    type(riverparameters), pointer :: river_parameters
    type(riverprognosticfields), pointer :: river_fields
      river_parameters => read_river_parameters(river_params_filename)
      river_fields => load_river_initial_values(hd_start_filename)
      global_prognostics = prognostics(using_lakes,river_parameters,river_fields)
      if (using_lakes) then
        if ( .not. present(lake_model_ctl_filename)) stop
        call init_lake_model(lake_model_ctl_filename,initial_spillover_to_rivers)
        call distribute_spillover(global_prognostics,initial_spillover_to_rivers)
        deallocate(initial_spillover_to_rivers)
      end if
      write_output = .true.
  end subroutine init_hd_model

  subroutine init_hd_model_for_testing(river_parameters,river_fields,using_lakes, &
                                       lake_parameters,initial_water_to_lake_centers, &
                                       initial_spillover_to_rivers)
    logical, intent(in) :: using_lakes
    type(riverparameters), intent(in) :: river_parameters
    type(lakeparameters), pointer, optional, intent(in) :: lake_parameters
    type(riverprognosticfields), pointer, optional, intent(inout) :: river_fields
    real, pointer, dimension(:,:),optional, intent(in) :: initial_spillover_to_rivers
    real, pointer, dimension(:,:),optional, intent(in) :: initial_water_to_lake_centers
      global_prognostics = prognostics(using_lakes,river_parameters,river_fields)
      if (using_lakes) then
        call init_lake_model_test(lake_parameters,initial_water_to_lake_centers)
        call distribute_spillover(global_prognostics,initial_spillover_to_rivers)
      end if
      write_output = .false.
  end subroutine init_hd_model_for_testing

  subroutine run_hd_model(timesteps,runoffs,drainages)
    integer, intent(in) :: timesteps
    real   ,dimension(:,:,:) :: runoffs
    real   ,dimension(:,:,:) :: drainages
    integer :: i
    do i = 1,timesteps
      call set_runoff_and_drainage(global_prognostics,runoffs(:,:,i),drainages(:,:,i))
      call run_hd(global_prognostics)
      if ((i == 1 .or. i == timesteps .or. mod(i,365) == 0) .and. write_output) then
        call write_river_flow_field(global_prognostics%river_parameters,&
                                    global_prognostics%river_fields%river_inflow,i)
        call write_lake_numbers_field_interface(i)
      end if
    end do
  end subroutine

  function get_global_prognostics() result(value)
    type(prognostics), pointer :: value
      value => global_prognostics
  end function get_global_prognostics

  subroutine clean_hd_model
    call global_prognostics%prognosticsdestructor()
  end subroutine clean_hd_model

end module latlon_hd_model_interface_mod
