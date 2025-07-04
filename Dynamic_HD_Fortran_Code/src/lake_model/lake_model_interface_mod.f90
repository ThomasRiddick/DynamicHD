module lake_model_interface_mod

use lake_model_mod
use lake_model_array_decoder_mod
use lake_model_input, only: config_lakes, load_lake_model_parameters, &
                               load_lake_initial_values
#ifdef USE_LOGGING
  use lake_logger_mod
#endif

implicit none

type lakeinterfaceprognosticfields
  real(dp), allocatable, dimension(_DIMS_)   :: water_from_lakes
  real(dp), allocatable, dimension(_DIMS_)   :: water_to_lakes
  !Lake water from ocean means negative water from lakes to ocean
  real(dp), allocatable, dimension(_DIMS_)   :: lake_water_from_ocean
  contains
    procedure :: initialiselakeinterfaceprognosticfields
    procedure :: lakeinterfaceprognosticfieldsdestructor
end type lakeinterfaceprognosticfields

interface lakeinterfaceprognosticfields
  procedure :: lakeinterfaceprognosticfieldsconstructor
end interface

type(lakemodelparameters), pointer ::  global_lake_model_parameters
type(lakemodelprognostics), pointer :: global_lake_model_prognostics
real(dp), private :: global_step_length
logical :: global_run_water_budget_check

contains

function lakeinterfaceprognosticfieldsconstructor(_NPOINTS_HD_) result(constructor)
  type(lakeinterfaceprognosticfields), pointer :: constructor
  _DEF_NPOINTS_HD_
    allocate(constructor)
    call constructor%initialiselakeinterfaceprognosticfields(_NPOINTS_HD_)
end function lakeinterfaceprognosticfieldsconstructor

subroutine  lakeinterfaceprognosticfieldsdestructor(this)
  class(lakeinterfaceprognosticfields) :: this
    deallocate(this%water_from_lakes)
    deallocate(this%water_to_lakes)
    deallocate(this%lake_water_from_ocean)
end subroutine  lakeinterfaceprognosticfieldsdestructor

subroutine initialiselakeinterfaceprognosticfields(this, &
                                                   _NPOINTS_HD_)
  class(lakeinterfaceprognosticfields) :: this
  _DEF_NPOINTS_HD_
    allocate(this%water_from_lakes(_NPOINTS_HD_))
    allocate(this%water_to_lakes(_NPOINTS_HD_))
    allocate(this%lake_water_from_ocean(_NPOINTS_HD_))
    this%water_from_lakes = 0.0_dp
    this%water_to_lakes = 0.0_dp
    this%lake_water_from_ocean = 0.0_dp
end subroutine initialiselakeinterfaceprognosticfields

subroutine init_lake_model_jsb(initial_spillover_to_rivers,cell_areas_from_jsbach, &
                               total_lake_restart_file_water_content,step_length)
  real(dp), dimension(_DIMS_), pointer, intent(out) :: initial_spillover_to_rivers
  real(dp), intent(out) :: total_lake_restart_file_water_content
  real(dp), dimension(_DIMS_), pointer, intent(in) :: cell_areas_from_jsbach
  real(dp), intent(in) :: step_length
  type(lakeparameterspointer), dimension(:), pointer :: lake_parameters_array
  real(dp), pointer, dimension(:) :: lake_parameters_as_array
  real(dp), pointer, dimension(_DIMS_) :: initial_water_to_lake_centers
  real(dp), pointer, dimension(:) :: lake_volumes
  real(dp), allocatable, dimension(_DIMS_) :: water_to_lakes_dummy
  real(dp), allocatable, dimension(_DIMS_) :: water_to_hd
  real(dp), allocatable, dimension(_DIMS_) :: lake_water_from_ocean
  character(len = 500) :: lake_model_ctl_filename
  logical :: use_binary_lake_mask
  _DEF_NPOINTS_HD_
  _DEF_NPOINTS_SURFACE_
    global_step_length = step_length
    _IF_USE_SINGLE_INDEX_
    !Not implemented
    _ELSE_
    _IF_USE_LONLAT_
    nlon_hd = 720
    nlat_hd = 360
    nlon_surface = 96
    nlat_surface = 48
    _ELSE_IF_NOT_USE_LONLAT_
    !Not implemented
    _END_IF_USE_LONLAT_
    _END_IF_USE_SINGLE_INDEX_
    lake_model_ctl_filename = "lake.ctl"
    call config_lakes(lake_model_ctl_filename,global_run_water_budget_check, &
                      use_binary_lake_mask)
    call load_lake_model_parameters(cell_areas_from_jsbach, &
                                    use_binary_lake_mask, &
                                    global_lake_model_parameters, &
                                    lake_parameters_as_array, &
                                    _NPOINTS_HD_, &
                                    _NPOINTS_SURFACE_)
    global_lake_model_prognostics => lakemodelprognostics(global_lake_model_parameters)
    call load_lake_initial_values(initial_water_to_lake_centers,&
                                  initial_spillover_to_rivers, &
                                  global_step_length, &
                                  global_lake_model_parameters%_NPOINTS_HD_)
    total_lake_restart_file_water_content = sum(initial_water_to_lake_centers) + &
                                            sum(initial_spillover_to_rivers)
    lake_parameters_array => &
      get_lake_parameters_from_array(lake_parameters_as_array, &
                                     global_lake_model_parameters%_NPOINTS_LAKE_, &
                                     global_lake_model_parameters%_NPOINTS_HD_)
    call create_lakes(global_lake_model_parameters, &
                      global_lake_model_prognostics, &
                      lake_parameters_array)
    call setup_lakes(global_lake_model_parameters,global_lake_model_prognostics, &
                     initial_water_to_lake_centers)
    allocate(water_to_lakes_dummy,mold=initial_spillover_to_rivers)
    water_to_lakes_dummy(:,:) = 0
    allocate(lake_water_from_ocean,mold=initial_spillover_to_rivers)
    allocate(water_to_hd,mold=initial_spillover_to_rivers)
    if (global_run_water_budget_check) then
      call check_water_budget(global_lake_model_prognostics, &
                              sum(initial_water_to_lake_centers))
    end if
    call run_lake_model_jsb(water_to_lakes_dummy,water_to_hd,lake_water_from_ocean)
    initial_spillover_to_rivers(:,:) = initial_spillover_to_rivers(:,:) + water_to_hd(:,:)/step_length
    water_to_hd(:,:) = 0.0_dp
    if (any(lake_water_from_ocean /= 0)) then
        write(*,*) "init_lake_water: unphysical lake evaporation during setup"
        stop
    end if
    lake_volumes => get_lake_volumes()
    write(*,'(A,ES15.7)') "Total initial water in lakes (m3): ",sum(lake_volumes)
    write(*,'(A,ES15.7)') "Total initial water spiltover to rivers from lakes (m3): ", &
                         sum(initial_spillover_to_rivers)
    write(*,'(A,ES15.7)') "Total initial water in lakes and spiltover to rivers (m3): ", &
                         sum(lake_volumes) + sum(initial_spillover_to_rivers)
    deallocate(lake_volumes)
    deallocate(water_to_lakes_dummy)
    deallocate(initial_water_to_lake_centers)
    deallocate(water_to_hd)
end subroutine init_lake_model_jsb

subroutine init_lake_model(lake_model_ctl_filename, &
                           cell_areas_on_surface_model_grid, &
                           initial_spillover_to_rivers, &
                           lake_interface_fields,step_length, &
                           _NPOINTS_HD_, &
                           _NPOINTS_SURFACE_)
  character(len = *), intent(in) :: lake_model_ctl_filename
  real(dp), pointer, dimension(_DIMS_), intent(in) :: cell_areas_on_surface_model_grid
  real(dp), pointer, dimension(_DIMS_),intent(out) :: initial_spillover_to_rivers
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
  real(dp), intent(in) :: step_length
  real(dp), pointer, dimension(:) :: lake_parameters_as_array
  type(lakeparameterspointer), dimension(:), pointer :: lake_parameters_array
  real(dp), pointer, dimension(_DIMS_) :: initial_water_to_lake_centers
  logical :: use_binary_lake_mask
  _DEF_NPOINTS_HD_ _INTENT_in_
  _DEF_NPOINTS_SURFACE_ _INTENT_in_
    call config_lakes(lake_model_ctl_filename,global_run_water_budget_check, &
                      use_binary_lake_mask)
    call load_lake_model_parameters(cell_areas_on_surface_model_grid, &
                                    use_binary_lake_mask, &
                                    global_lake_model_parameters, &
                                    lake_parameters_as_array, &
                                    _NPOINTS_HD_, &
                                    _NPOINTS_SURFACE_)
    global_lake_model_prognostics => lakemodelprognostics(global_lake_model_parameters)
    global_step_length = step_length
    call load_lake_initial_values(initial_water_to_lake_centers,&
                                  initial_spillover_to_rivers, &
                                  global_step_length, &
                                  global_lake_model_parameters%_NPOINTS_HD_)
    lake_parameters_array => &
      get_lake_parameters_from_array(lake_parameters_as_array, &
                                     global_lake_model_parameters%_NPOINTS_LAKE_, &
                                     global_lake_model_parameters%_NPOINTS_HD_)
    call create_lakes(global_lake_model_parameters, &
                      global_lake_model_prognostics, &
                      lake_parameters_array)
    call setup_lakes(global_lake_model_parameters,global_lake_model_prognostics, &
                     initial_water_to_lake_centers)
    call run_lakes(global_lake_model_parameters,global_lake_model_prognostics)
    if (global_run_water_budget_check) then
      call check_water_budget(global_lake_model_prognostics, &
                              sum(initial_water_to_lake_centers))
    end if
    global_lake_model_prognostics%water_to_hd(_DIMS_) = &
      global_lake_model_prognostics%water_to_hd(_DIMS_)/global_step_length
    lake_interface_fields%water_from_lakes(_DIMS_) = global_lake_model_prognostics%water_to_hd(_DIMS_)
    deallocate(initial_water_to_lake_centers)
end subroutine init_lake_model

subroutine init_lake_model_test(lake_model_parameters,lake_parameters_as_array, &
                                initial_water_to_lake_centers, &
                                lake_interface_fields,step_length)
  type(lakemodelparameters), pointer :: lake_model_parameters
  real(dp), pointer, dimension(:), intent(in) :: lake_parameters_as_array
  real(dp), pointer, dimension(_DIMS_), intent(in) :: initial_water_to_lake_centers
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
  type(lakeparameterspointer), dimension(:), pointer :: lake_parameters_array
  real(dp) :: step_length
    global_lake_model_parameters => lake_model_parameters
    ! if (associated(global_lake_model_prognostics)) then
    !     call global_lake_model_prognostics%lakemodelprognosticsdestructor()
    !     deallocate(global_lake_model_prognostics)
    ! end if
    global_lake_model_prognostics => lakemodelprognostics(global_lake_model_parameters)
    global_step_length = step_length
    lake_parameters_array => &
      get_lake_parameters_from_array(lake_parameters_as_array, &
                                     global_lake_model_parameters%_NPOINTS_LAKE_, &
                                     global_lake_model_parameters%_NPOINTS_HD_)
    call create_lakes(global_lake_model_parameters, &
                      global_lake_model_prognostics, &
                      lake_parameters_array)
    call setup_lakes(global_lake_model_parameters,global_lake_model_prognostics, &
                     initial_water_to_lake_centers)
    call run_lakes(global_lake_model_parameters,global_lake_model_prognostics)
    call check_water_budget(global_lake_model_prognostics, &
                            sum(initial_water_to_lake_centers))
    global_run_water_budget_check = .true.
    global_lake_model_prognostics%water_to_hd(_DIMS_) = &
      global_lake_model_prognostics%water_to_hd(_DIMS_)/global_step_length
    lake_interface_fields%water_from_lakes(_DIMS_) = global_lake_model_prognostics%water_to_hd(_DIMS_)
end subroutine init_lake_model_test

subroutine clean_lake_model()
#ifdef USE_LOGGING
  call delete_logger
#endif
  call clean_lake_model_prognostics(global_lake_model_prognostics)
  call clean_lake_model_parameters(global_lake_model_parameters)
  deallocate(global_lake_model_parameters)
  deallocate(global_lake_model_prognostics)
end subroutine clean_lake_model

subroutine run_lake_model(lake_interface_fields)
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
    global_lake_model_prognostics%water_to_lakes(_DIMS_) = &
      lake_interface_fields%water_to_lakes(_DIMS_)*global_step_length
    call run_lakes(global_lake_model_parameters,global_lake_model_prognostics)
    if (global_run_water_budget_check) then
      call check_water_budget(global_lake_model_prognostics)
    end if
    global_lake_model_prognostics%water_to_hd(_DIMS_) = &
      global_lake_model_prognostics%water_to_hd(_DIMS_)/global_step_length
    lake_interface_fields%water_from_lakes(_DIMS_) = global_lake_model_prognostics%water_to_hd(_DIMS_)
    global_lake_model_prognostics%lake_water_from_ocean(_DIMS_) = &
      global_lake_model_prognostics%lake_water_from_ocean(_DIMS_)/global_step_length
    lake_interface_fields%lake_water_from_ocean(_DIMS_) = &
      global_lake_model_prognostics%lake_water_from_ocean(_DIMS_)
end subroutine run_lake_model

subroutine run_lake_model_jsb(water_to_lakes_in,water_to_hd_out,lake_water_from_ocean)
  real(dp), allocatable, dimension(_DIMS_), intent(in)   :: water_to_lakes_in
  real(dp), allocatable, dimension(_DIMS_), intent(inout)  :: water_to_hd_out
  real(dp), allocatable, dimension(:,:), intent(inout)  :: lake_water_from_ocean
    global_lake_model_prognostics%water_to_lakes(_DIMS_) = water_to_lakes_in(_DIMS_)
    call run_lakes(global_lake_model_parameters,global_lake_model_prognostics)
    if (global_run_water_budget_check) then
      call check_water_budget(global_lake_model_prognostics)
    end if
    water_to_hd_out(_DIMS_) = global_lake_model_prognostics%water_to_hd(_DIMS_)
    lake_water_from_ocean(_DIMS_) = global_lake_model_prognostics%lake_water_from_ocean(_DIMS_)
end subroutine run_lake_model_jsb

! subroutine write_lake_numbers_field_interface(working_directory,timestep)
!   integer :: timestep
!   character(len = *), intent(in) :: working_directory
!     call write_lake_numbers_field(working_directory,global_lake_model_parameters,&
!                                   timestep)
! end subroutine write_lake_numbers_field_interface

subroutine write_diagnostic_lake_volumes_interface(working_directory,timestep)
  integer :: timestep
  character(len = *), intent(in) :: working_directory
    call write_diagnostic_lake_volumes(working_directory, &
                                       global_lake_model_parameters, &
                                       global_lake_model_prognostics, &
                                       timestep)
end subroutine write_diagnostic_lake_volumes_interface

subroutine write_lake_fractions_interface(working_directory,timestep)
  integer :: timestep
  character(len = *), intent(in) :: working_directory
    call write_lake_fractions(working_directory, &
                              global_lake_model_parameters, &
                              global_lake_model_prognostics, &
                              timestep)
end subroutine write_lake_fractions_interface

subroutine write_lake_volumes_interface(lake_volumes_filename)
  character(len = *), intent(in) :: lake_volumes_filename
  call write_lake_volumes(lake_volumes_filename, &
                          global_lake_model_parameters, &
                          global_lake_model_prognostics)
end subroutine write_lake_volumes_interface

subroutine write_binary_lake_mask_and_adjusted_lake_fraction_interface(binary_lake_mask_filename)
  character(len = *), intent(in) :: binary_lake_mask_filename
  call write_binary_lake_mask_and_adjusted_lake_fraction(binary_lake_mask_filename, &
                                                         global_lake_model_parameters, &
                                                         global_lake_model_prognostics)
end subroutine write_binary_lake_mask_and_adjusted_lake_fraction_interface


function get_lake_model_prognostics() result(value)
  type(lakemodelprognostics), pointer :: value
    value => global_lake_model_prognostics
end function get_lake_model_prognostics

subroutine get_lake_fraction(lake_fraction)
  real(dp), dimension(_DIMS_), allocatable, intent(inout) :: lake_fraction
    call calculate_lake_fraction_on_surface_grid(global_lake_model_parameters, &
                                                 global_lake_model_prognostics, &
                                                 lake_fraction)
end subroutine get_lake_fraction

subroutine set_lake_evaporation_for_testing_interface(lake_evaporation)
  real(dp), allocatable, dimension(_DIMS_), intent(in) :: lake_evaporation
    call set_lake_evaporation_for_testing(global_lake_model_parameters, &
                                          global_lake_model_prognostics, &
                                          lake_evaporation)
end subroutine set_lake_evaporation_for_testing_interface

subroutine set_lake_evaporation_interface(height_of_water_evaporated)
  real(dp), allocatable, dimension(_DIMS_), intent(in) :: height_of_water_evaporated
    call set_lake_evaporation(global_lake_model_parameters, &
                              global_lake_model_prognostics, &
                              height_of_water_evaporated)
end subroutine set_lake_evaporation_interface

_IF_USE_SINGLE_INDEX_

_ELSE_

function get_surface_model_nlat() result(nlat)
  integer :: nlat
    nlat = global_lake_model_parameters%nlat_surface
end function get_surface_model_nlat

function get_surface_model_nlon() result(nlon)
  integer :: nlon
    nlon = global_lake_model_parameters%nlon_surface
end function get_surface_model_nlon

_END_IF_USE_SINGLE_INDEX_

function get_lake_volumes() result(lake_volumes)
  real(dp), pointer, dimension(:) :: lake_volumes
    lake_volumes => get_lake_volume_list(global_lake_model_prognostics)
end function get_lake_volumes

end module lake_model_interface_mod
