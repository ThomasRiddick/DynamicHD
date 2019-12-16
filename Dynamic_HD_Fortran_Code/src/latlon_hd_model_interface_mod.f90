module latlon_hd_model_interface_mod

  use latlon_hd_model_mod
  use latlon_hd_model_io_mod

  implicit none

  type(prognostics), target :: global_prognostics

  contains

  subroutine init_hd_model(river_params_filename,hd_start_filename,lake_model_ctl_filename,&
                           using_lakes)
    logical, intent(in) :: using_lakes
    character(len = *) :: river_params_filename
    character(len = *) :: hd_start_filename
    character(len = *) :: lake_model_ctl_filename
    real, allocatable, dimension(:,:) :: initial_spillover_to_rivers
    type(riverparameters), pointer :: river_parameters
    type(riverprognosticfields), pointer :: river_fields
      river_parameters => read_river_parameters(river_params_filename)
      river_fields => load_river_initial_values(hd_start_filename)
      global_prognostics = prognostics(using_lakes,river_parameters,river_fields)
      if (using_lakes) then
        call init_lake_model(lake_model_ctl_filename,initial_spillover_to_rivers)
        call distribute_spillover(global_prognostics,initial_spillover_to_rivers)
      end if
  end subroutine init_hd_model

  subroutine init_hd_model_for_testing(river_parameters,river_fields,using_lakes, &
                                       lake_parameters,initial_water_to_lake_centers, &
                                       initial_spillover_to_rivers)
    logical, intent(in) :: using_lakes
    type(riverparameters), intent(in) :: river_parameters
    type(lakeparameters), pointer, optional, intent(in) :: lake_parameters
    type(riverprognosticfields),optional, intent(inout) :: river_fields
    real, allocatable, dimension(:,:),optional, intent(in) :: initial_spillover_to_rivers
    real, allocatable, dimension(:,:),optional, intent(in) :: initial_water_to_lake_centers
      global_prognostics = prognostics(using_lakes,river_parameters,river_fields)
      if (using_lakes) then
        call init_lake_model_test(lake_parameters,initial_water_to_lake_centers)
        call distribute_spillover(global_prognostics,initial_spillover_to_rivers)
      end if
  end subroutine init_hd_model_for_testing

  subroutine run_hd_model(timesteps,runoffs,drainages)
    integer, intent(in) :: timesteps
    real   ,dimension(:,:,:) :: runoffs
    real   ,dimension(:,:,:) :: drainages
    integer :: i
    do i = 1,timesteps
      call set_runoff_and_drainage(global_prognostics,runoffs(:,:,i),drainages(:,:,i))
      call run_hd(global_prognostics)
    end do
  end subroutine

  function get_global_prognostics() result(value)
    type(prognostics), pointer :: value
      value => global_prognostics
  end function get_global_prognostics

  subroutine clean_hd_model
    call global_prognostics%prognosticsdestructor()
  end subroutine clean_hd_model

  subroutine add_offset(array,offset,exceptions)
    integer, dimension(:,:), intent(inout) :: array
    integer, intent(in) :: offset
    integer, dimension(:), optional, intent(in) :: exceptions
    logical, dimension(:,:), allocatable :: exception_mask
    integer :: exception
    integer :: i
      allocate(exception_mask(size(array,1),size(array,2)))
      exception_mask = .False.
      if(present(exceptions)) then
        do i = 1,size(exceptions)
          exception = exceptions(i)
          where(array == exception)
            exception_mask=.True.
          end where
        end do
      end if
      where(.not. exception_mask)
        array = array + offset
      end where
  end subroutine

end module latlon_hd_model_interface_mod
