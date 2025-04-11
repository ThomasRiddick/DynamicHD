module l2_lake_model_interface_switcher_mod

use l2_lake_model_interface_mod, only: init_lake_model_test_orig => init_lake_model_test, &
                                       init_lake_model_orig => init_lake_model, &
                                       clean_lake_model_orig => clean_lake_model, &
                                       run_lake_model_orig => run_lake_model, &
                                       get_lake_model_prognostics_orig => &
                                       get_lake_model_prognostics, &
                                       get_surface_model_nlat_orig => get_surface_model_nlat, &
                                       get_surface_model_nlon_orig => get_surface_model_nlon, &
                                       get_lake_volumes_orig => get_lake_volumes, &
                                       get_lake_fraction_orig => get_lake_fraction, &
                                       set_lake_evaporation_for_testing_interface_orig => &
                                       set_lake_evaporation_for_testing_interface, &
                                       set_lake_evaporation_interface_orig => &
                                       set_lake_evaporation_interface, &
                                       lakeinterfaceprognosticfields
use l2_lake_model_mod,          only:  lakemodelparameters,lakemodelprognostics, &
                                       lakemodelparametersconstructor_orig => &
                                       lakemodelparametersconstructor, &
                                       dp, filling_lake_type, overflowing_lake_type, &
                                       subsumed_lake_type, &
                                       calculate_diagnostic_lake_volumes_field_orig => &
                                       calculate_diagnostic_lake_volumes_field, &
                                       calculate_lake_fraction_on_surface_grid

implicit none

interface transposedlakemodelparameters
  procedure :: transposed_lakemodelparametersconstructor
end interface

contains

subroutine init_lake_model_test(lake_model_parameters,lake_parameters_as_array, &
                                initial_water_to_lake_centers, &
                                lake_interface_fields,step_length)
  type(lakemodelparameters), pointer, intent(in) :: lake_model_parameters
  real(dp), pointer, dimension(:), intent(in) :: lake_parameters_as_array
  real(dp), pointer, dimension(:,:), intent(in) :: initial_water_to_lake_centers
  type(lakeinterfaceprognosticfields), pointer, intent(inout) :: lake_interface_fields
  real(dp), pointer, dimension(:,:)  :: transposed_initial_water_to_lake_centers
  type(lakeinterfaceprognosticfields), pointer :: transposed_lake_interface_fields
  real(dp) :: step_length
  integer, dimension(2) :: array_shape
        array_shape = shape(initial_water_to_lake_centers)
        allocate(transposed_initial_water_to_lake_centers(array_shape(2),&
                                                          array_shape(1)))
        transposed_lake_interface_fields => transpose_lake_interface_fields(lake_interface_fields)
        transposed_initial_water_to_lake_centers = transpose(initial_water_to_lake_centers)
        call init_lake_model_test_orig(lake_model_parameters,lake_parameters_as_array, &
                                       transposed_initial_water_to_lake_centers, &
                                       transposed_lake_interface_fields,step_length)
        lake_interface_fields => transpose_lake_interface_fields(transposed_lake_interface_fields)
end subroutine init_lake_model_test

subroutine clean_lake_model()
        call clean_lake_model_orig()
end subroutine clean_lake_model

subroutine run_lake_model(lake_interface_fields)
  type(lakeinterfaceprognosticfields), pointer, intent(inout) :: lake_interface_fields
  type(lakeinterfaceprognosticfields), pointer :: transposed_lake_interface_fields
        transposed_lake_interface_fields => transpose_lake_interface_fields(lake_interface_fields)
        call run_lake_model_orig(transposed_lake_interface_fields)
        lake_interface_fields => transpose_lake_interface_fields(transposed_lake_interface_fields)
end subroutine run_lake_model

subroutine write_lake_numbers_field_interface(working_directory,timestep)
  integer :: timestep
  character(len = *), intent(in) :: working_directory
  integer :: i
    if (timestep == 0) i = 0
    if (working_directory == "") i = 0
    write(*,*) "write_lake_numbers_field_interface has been disabled"
end subroutine write_lake_numbers_field_interface

subroutine write_diagnostic_lake_volumes_interface(working_directory,timestep)
  integer :: timestep
  character(len = *), intent(in) :: working_directory
  integer :: i
    if (timestep == 0) i = 0
    if (working_directory == "") i = 0
    write(*,*) "write_diagnostic_lake_volumes_interface has been disabled"
end subroutine write_diagnostic_lake_volumes_interface

function get_lake_model_prognostics() result(value)
  type(lakemodelprognostics), pointer :: value
        !value =>  transpose_lake_prognostics(get_lake_prognostics_orig())
        value => get_lake_model_prognostics_orig()
end function get_lake_model_prognostics

function get_surface_model_nlat() result(nlat)
  integer :: nlat
    nlat = get_surface_model_nlon_orig()
end function get_surface_model_nlat

function get_surface_model_nlon() result(nlon)
  integer :: nlon
    nlon = get_surface_model_nlat_orig()
end function get_surface_model_nlon

function get_lake_volumes() result(lake_volumes)
  real(dp), pointer, dimension(:) :: lake_volumes
    lake_volumes => get_lake_volumes_orig()
end function get_lake_volumes

subroutine get_lake_fraction(lake_fraction)
  real(dp), dimension(:,:) :: lake_fraction
  real(dp), allocatable, dimension(:,:) :: lake_fraction_orig
  integer, dimension(2) :: array_shape
        array_shape = shape(lake_fraction)
        allocate(lake_fraction_orig(array_shape(2),&
                                    array_shape(1)))
        call get_lake_fraction_orig(lake_fraction_orig)
        lake_fraction = transpose(lake_fraction_orig)
end subroutine get_lake_fraction

subroutine set_lake_evaporation_for_testing_interface(lake_evaporation)
  real(dp), allocatable, dimension(:,:), intent(in) :: lake_evaporation
  real(dp), allocatable, dimension(:,:) :: lake_evaporation_transposed
  integer, dimension(2) :: array_shape
        array_shape = shape(lake_evaporation)
        allocate(lake_evaporation_transposed(array_shape(2),&
                                             array_shape(1)))
        lake_evaporation_transposed = transpose(lake_evaporation)
        call set_lake_evaporation_for_testing_interface_orig(lake_evaporation_transposed)
end subroutine set_lake_evaporation_for_testing_interface

subroutine set_lake_evaporation_interface(height_of_water_evaporated)
  real(dp), allocatable, dimension(:,:), intent(in) :: height_of_water_evaporated
  real(dp), allocatable, dimension(:,:) :: height_of_water_evaporated_transposed
  integer, dimension(2) :: array_shape
        array_shape = shape(height_of_water_evaporated)
        allocate(height_of_water_evaporated_transposed(array_shape(2),&
                                                       array_shape(1)))
        height_of_water_evaporated_transposed = transpose(height_of_water_evaporated)
        call set_lake_evaporation_interface_orig(height_of_water_evaporated_transposed)
end subroutine set_lake_evaporation_interface

function transposed_lakemodelparametersconstructor( &
    corresponding_surface_cell_lat_index, &
    corresponding_surface_cell_lon_index, &
    cell_areas_on_surface_model_grid, &
    number_of_lakes, &
    is_lake, &
    binary_lake_mask, &
    nlat_hd,nlon_hd, &
    nlat_lake,nlon_lake, &
    nlat_surface,nlon_surface, &
    lake_retention_constant, &
    minimum_lake_volume_threshold) &
      result(constructor)
  logical, pointer, dimension(:,:), intent(in) :: is_lake
  integer, dimension(:,:), pointer, intent(in) :: corresponding_surface_cell_lat_index
  integer, dimension(:,:), pointer, intent(in) :: corresponding_surface_cell_lon_index
  real(dp), dimension(:,:), pointer, intent(in) :: cell_areas_on_surface_model_grid
  logical, dimension(:,:), pointer, intent(in) :: binary_lake_mask
  integer, intent(in) :: number_of_lakes
  integer, intent(in) :: nlat_hd,nlon_hd
  integer, intent(in) :: nlat_lake,nlon_lake
  integer, intent(in) :: nlat_surface,nlon_surface
  real(dp), intent(inout), optional :: lake_retention_constant
  real(dp), intent(inout), optional :: minimum_lake_volume_threshold
  logical, dimension(:,:), pointer :: transposed_is_lake
  integer, dimension(:,:), pointer :: transposed_corresponding_surface_cell_lat_index
  integer, dimension(:,:), pointer :: transposed_corresponding_surface_cell_lon_index
  real(dp), dimension(:,:), pointer :: transposed_cell_areas_on_surface_model_grid
  logical, dimension(:,:), pointer :: transposed_binary_lake_mask
  type(lakemodelparameters), pointer :: constructor
    allocate(transposed_is_lake(nlon_lake,nlat_lake))
    transposed_is_lake = transpose(is_lake)
    allocate(transposed_corresponding_surface_cell_lat_index(nlon_lake,nlat_lake))
    transposed_corresponding_surface_cell_lat_index = transpose(corresponding_surface_cell_lat_index)
    allocate(transposed_corresponding_surface_cell_lon_index(nlon_lake,nlat_lake))
    transposed_corresponding_surface_cell_lon_index = transpose(corresponding_surface_cell_lon_index)
    allocate(transposed_cell_areas_on_surface_model_grid(nlon_surface,nlat_surface))
    transposed_cell_areas_on_surface_model_grid = transpose(cell_areas_on_surface_model_grid)
    allocate(transposed_binary_lake_mask(nlon_surface,nlat_surface))
    transposed_binary_lake_mask = transpose(binary_lake_mask)
    constructor => lakemodelparametersconstructor_orig( &
      transposed_corresponding_surface_cell_lon_index, &
      transposed_corresponding_surface_cell_lat_index, &
      transposed_cell_areas_on_surface_model_grid, &
      number_of_lakes, &
      transposed_is_lake, &
      transposed_binary_lake_mask, &
      nlon_hd,nlat_hd, &
      nlon_lake,nlat_lake, &
      nlon_surface,nlat_surface, &
      lake_retention_constant, &
      minimum_lake_volume_threshold)
end function transposed_lakemodelparametersconstructor

function transpose_lake_interface_fields(lake_interface_fields) result(transposed_lake_interface_fields)
  type(lakeinterfaceprognosticfields), pointer :: lake_interface_fields
  type(lakeinterfaceprognosticfields), pointer :: transposed_lake_interface_fields
  integer, dimension(2) :: array_shape
    array_shape = shape(lake_interface_fields%water_from_lakes)
    transposed_lake_interface_fields => lakeinterfaceprognosticfields(array_shape(2),&
                                                                      array_shape(1))
    transposed_lake_interface_fields%water_from_lakes = transpose(lake_interface_fields%water_from_lakes)
    transposed_lake_interface_fields%water_to_lakes = transpose(lake_interface_fields%water_to_lakes)
    transposed_lake_interface_fields%lake_water_from_ocean = transpose(lake_interface_fields%lake_water_from_ocean)
end function transpose_lake_interface_fields

function calculate_diagnostic_lake_volumes_field(lake_model_parameters, &
                                                 lake_model_prognostics) result(diagnostic_lake_volumes_transposed)
  type(lakemodelparameters), pointer, intent(in) :: lake_model_parameters
  type(lakemodelprognostics), pointer, intent(inout) :: lake_model_prognostics
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volumes
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volumes_transposed
  integer, dimension(2) :: array_shape
    diagnostic_lake_volumes => calculate_diagnostic_lake_volumes_field_orig(lake_model_parameters,&
                                                                            lake_model_prognostics)
    array_shape = shape(diagnostic_lake_volumes)
    allocate(diagnostic_lake_volumes_transposed(array_shape(2),&
                                                array_shape(1)))
    diagnostic_lake_volumes_transposed = transpose(diagnostic_lake_volumes)
end function  calculate_diagnostic_lake_volumes_field

end module l2_lake_model_interface_switcher_mod
