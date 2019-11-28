module latlon_hd_model_interface_mod

  use latlon_hd_model_mod
  use latlon_hd_model_io_mod

  implicit none

  type(prognostics) :: global_prognostics

  subroutine init_hd_model(river_params_filename,hd_start_filename,lake_model_ctl_filename,&
                           using_lakes)
    logical :: using_lakes
    character(len = *) :: river_params_filename
    character(len = *) :: hd_start_filename
    character(len = *) :: lake_model_ctl_filename
      using_lakes = .false.
      river_parameters = read_river_parameters(river_params_filename)
      river_prognostic_fields = load_river_initial_values(hd_start_filename)
      global_prognostics = prognostics(using_lakes,river_parameters,river_prognostic_fields)
      if (using_lakes) then
        call init_lake_model(lake_model_ctl_filename)
      end if
  end subroutine init_hd_model

  subroutine init_hd_model_for_testing

  end subroutine init_hd_model_for_testing

  subroutine run_hd_model(timesteps)
    integer, intent(int) :: timesteps
    integer :: i
    do i = 1,timesteps
      call run_hd(global_prognostics)
    end do
  end subroutine

end module latlon_hd_model_interface_mod
