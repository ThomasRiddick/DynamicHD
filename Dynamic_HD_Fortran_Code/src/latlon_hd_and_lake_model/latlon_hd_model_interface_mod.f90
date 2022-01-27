module latlon_hd_model_interface_mod

  use latlon_hd_model_mod
  use latlon_hd_model_io_mod

  implicit none

  type(prognostics), pointer :: global_prognostics
  logical :: write_output
  real(dp) :: global_step_length = 86400.0_dp

  contains

  subroutine init_hd_model(river_params_filename,hd_start_filename,using_lakes,&
                           lake_model_ctl_filename)
    logical, intent(in) :: using_lakes
    character(len = *) :: river_params_filename
    character(len = *) :: hd_start_filename
    character(len = *), optional :: lake_model_ctl_filename
    real(dp), pointer, dimension(:,:) :: initial_spillover_to_rivers
    type(riverparameters), pointer :: river_parameters
    type(riverprognosticfields), pointer :: river_fields
      river_parameters => read_river_parameters(river_params_filename)
      river_fields => load_river_initial_values(hd_start_filename)
      global_prognostics => prognostics(using_lakes,river_parameters,river_fields)
      if (using_lakes) then
        if ( .not. present(lake_model_ctl_filename)) stop
        call init_lake_model(lake_model_ctl_filename,initial_spillover_to_rivers, &
                             global_prognostics%lake_interface_fields,global_step_length)
        initial_spillover_to_rivers(:,:) = initial_spillover_to_rivers(:,:) + &
          global_prognostics%lake_interface_fields%water_from_lakes(:,:)
        global_prognostics%lake_interface_fields%water_from_lakes(:,:) = 0.0_dp
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
    real(dp), pointer, dimension(:,:) :: initial_spillover_to_rivers_local
    real(dp), pointer, dimension(:,:) :: initial_water_to_lake_centers_local
    real(dp), pointer, dimension(:,:),optional, intent(in) :: initial_spillover_to_rivers
    real(dp), pointer, dimension(:,:),optional, intent(in) :: initial_water_to_lake_centers
      if(associated(global_prognostics)) then
        if (using_lakes) then
          call global_prognostics%lake_interface_fields%&
               &lakeinterfaceprognosticfieldsdestructor()
          deallocate(global_prognostics%lake_interface_fields)
        end if
        call global_prognostics%river_diagnostic_fields%riverdiagnosticfieldsdestructor()
        deallocate(global_prognostics%river_diagnostic_fields)
        call global_prognostics%river_diagnostic_output_fields%&
                                &riverdiagnosticoutputfieldsdestructor()
        deallocate(global_prognostics%river_diagnostic_output_fields)
        deallocate(global_prognostics)
      end if
      global_prognostics => prognostics(using_lakes,river_parameters,river_fields)
      if (using_lakes) then
        allocate(initial_water_to_lake_centers_local(lake_parameters%nlat, &
                                                     lake_parameters%nlon))
        if (present(initial_water_to_lake_centers)) then
          initial_water_to_lake_centers_local(:,:) = &
            initial_water_to_lake_centers(:,:)
        else
          initial_water_to_lake_centers_local(:,:) = 0.0_dp
        end if
        call init_lake_model_test(lake_parameters,initial_water_to_lake_centers_local, &
                                  global_prognostics%lake_interface_fields, &
                                  global_step_length)
        deallocate(initial_water_to_lake_centers_local)
        allocate(initial_spillover_to_rivers_local(river_parameters%nlat, &
                                                   river_parameters%nlon))
        if (present(initial_spillover_to_rivers)) then
          initial_spillover_to_rivers_local(:,:) = &
            initial_spillover_to_rivers(:,:)/global_step_length
        else
          initial_spillover_to_rivers_local(:,:) = 0.0_dp
        end if
        initial_spillover_to_rivers_local(:,:) = initial_spillover_to_rivers_local(:,:) + &
          global_prognostics%lake_interface_fields%water_from_lakes(:,:)
        global_prognostics%lake_interface_fields%water_from_lakes(:,:) = 0.0_dp
        call distribute_spillover(global_prognostics,initial_spillover_to_rivers_local)
        deallocate(initial_spillover_to_rivers_local)
      end if
      write_output = .false.
  end subroutine init_hd_model_for_testing

  subroutine run_hd_model(timesteps,runoffs,drainages,lake_evaporations,&
                          use_realistic_surface_coupling_in,working_directory, &
                          lake_volumes_for_all_timesteps)
    integer, intent(in) :: timesteps
    real(dp)   ,dimension(:,:,:) :: runoffs
    real(dp)   ,dimension(:,:,:) :: drainages
    real(dp)   ,dimension(:,:,:), optional :: lake_evaporations
    logical, intent(in), optional :: use_realistic_surface_coupling_in
    real(dp)   ,dimension(:,:,:), allocatable :: lake_evaporations_local
    character(len = *), intent(in),optional :: working_directory
    real(dp)   ,dimension(:,:), allocatable :: runoff
    real(dp)   ,dimension(:,:), allocatable :: drainage
    real(dp)   ,dimension(:,:), allocatable :: evaporation
    real(dp)   ,dimension(:,:), allocatable :: lake_fraction_adjusted_evaporation
    real(dp)   ,dimension(:,:), pointer     :: lake_fractions
    real(dp)   ,dimension(:,:), optional, pointer :: lake_volumes_for_all_timesteps
    real(dp)   ,dimension(:), pointer :: lake_volumes
    logical :: use_realistic_surface_coupling
    integer :: i
    character(len = max_name_length) :: working_directory_local
      if (present(working_directory)) then
        working_directory_local = working_directory
      else
        working_directory_local = ""
      end if
      if (present(use_realistic_surface_coupling_in)) then
        use_realistic_surface_coupling = use_realistic_surface_coupling_in
      else
        use_realistic_surface_coupling = .false.
      end if
      if (present(lake_evaporations)) then
        allocate(lake_evaporations_local,mold=lake_evaporations)
        lake_evaporations_local(:,:,:) = lake_evaporations(:,:,:)
      else if (global_prognostics%using_lakes) then
          allocate(lake_evaporations_local(get_surface_model_nlat_interface(), &
                                           get_surface_model_nlon_interface(), &
                                           size(runoffs,3)))
          lake_evaporations_local(:,:,:) = 0.0_dp
      end if
      allocate(runoff(global_prognostics%river_parameters%nlat,&
                      global_prognostics%river_parameters%nlon))
      allocate(drainage(global_prognostics%river_parameters%nlat,&
                        global_prognostics%river_parameters%nlon))
      if (global_prognostics%using_lakes) then
        allocate(evaporation(get_surface_model_nlat_interface(), &
                             get_surface_model_nlon_interface()))
        allocate(lake_fraction_adjusted_evaporation(get_surface_model_nlat_interface(), &
                                                    get_surface_model_nlon_interface()))
      end if
      do i = 1,timesteps
        runoff(:,:) = runoffs(:,:,i)
        drainage(:,:) = drainages(:,:,i)
        if (global_prognostics%using_lakes) then
          evaporation(:,:) = lake_evaporations_local(:,:,i)
          if (use_realistic_surface_coupling) then
            lake_fractions => get_lake_fractions_interface()
            lake_fraction_adjusted_evaporation(:,:) = evaporation(:,:) * lake_fractions(:,:)
            call set_evaporation_to_lakes_interface(lake_fraction_adjusted_evaporation)
            deallocate(lake_fractions)
          else
            call set_evaporation_to_lakes_for_testing_interface(evaporation)
          end if
        end if
        call set_runoff_and_drainage(global_prognostics,runoff,drainage)
        call run_hd(global_prognostics)
        if (present(lake_volumes_for_all_timesteps)) then
          lake_volumes => get_lake_volumes()
          lake_volumes_for_all_timesteps(:,i) = lake_volumes(:)
          deallocate(lake_volumes)
        end if
        if ((i == 1 .or. i == timesteps .or. mod(i,365) == 0) .and. write_output) then
          call write_river_flow_field(working_directory_local, &
                                      global_prognostics%river_parameters,&
                                      global_prognostics%river_fields%river_inflow,i)
          call write_diagnostic_lake_volumes_interface(working_directory_local,i)
        end if
      end do
      if (global_prognostics%using_lakes) then
          deallocate(lake_evaporations_local)
          deallocate(lake_fraction_adjusted_evaporation)
          deallocate(evaporation)
      end if
      deallocate(runoff)
      deallocate(drainage)
  end subroutine

  function get_global_prognostics() result(value)
    type(prognostics), pointer :: value
      value => global_prognostics
  end function get_global_prognostics

  subroutine clean_hd_model
    call global_prognostics%prognosticsdestructor()
    deallocate(global_prognostics)
  end subroutine clean_hd_model

end module latlon_hd_model_interface_mod
