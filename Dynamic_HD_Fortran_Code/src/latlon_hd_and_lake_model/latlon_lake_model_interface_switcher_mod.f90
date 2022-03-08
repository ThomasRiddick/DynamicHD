module latlon_lake_model_interface_switcher_mod

use latlon_lake_model_interface_switched_mod, only: init_lake_model_test_orig => init_lake_model_test, &
                                                    init_lake_model_orig => init_lake_model, &
                                                    clean_lake_model_orig => clean_lake_model, &
                                                    run_lake_model_orig => run_lake_model_test, &
                                                    get_lake_prognostics_orig => &
                                                    get_lake_prognostics, &
                                                    get_lake_fields_orig => get_lake_fields, &
                                                    get_surface_model_nlat_orig => get_surface_model_nlat, &
                                                    get_surface_model_nlon_orig => get_surface_model_nlon, &
                                                    get_lake_volumes_orig => get_lake_volumes, &
                                                    get_lake_fraction_orig => get_lake_fraction, &
                                                    set_lake_evaporation_for_testing_interface_orig => &
                                                    set_lake_evaporation_for_testing_interface, &
                                                    set_lake_evaporation_interface_orig => &
                                                    set_lake_evaporation_interface, &
                                                    lakeinterfaceprognosticfields
use latlon_lake_model_switched_mod,          only:  lakeparameters, lakefields, lakeprognostics, &
                                                    dp, lakepointer, filling_lake_type, &
                                                    overflowing_lake_type, subsumed_lake_type, &
                                                    calculate_diagnostic_lake_volumes_orig => &
                                                    calculate_diagnostic_lake_volumes, &
                                                    calculate_lake_fraction_on_surface_grid
use precision_mod
use parameters_mod

implicit none

contains

subroutine init_lake_model(lake_model_ctl_filename,initial_spillover_to_rivers, &
                           lake_interface_fields,step_length)
  character(len = *) :: lake_model_ctl_filename
  real(dp), pointer, dimension(:,:),intent(out) :: initial_spillover_to_rivers
  real(dp), allocatable, target, dimension(:,:) :: initial_spillover_to_rivers_alloc
  type(lakeinterfaceprognosticfields), intent(inout) :: lake_interface_fields
  real(dp) :: step_length
  integer :: i
      allocate(initial_spillover_to_rivers_alloc,mold=lake_interface_fields%water_from_lakes)
      lake_model_ctl_filename = ""
      if (lake_interface_fields%water_to_lakes(0,0) == 0.0_dp) i = 0
      if(step_length == 0) i = 0
      call init_lake_model_orig(initial_spillover_to_rivers_alloc)
      initial_spillover_to_rivers(:,:) = initial_spillover_to_rivers_alloc(:,:)
end subroutine

subroutine init_lake_model_test(lake_parameters,initial_water_to_lake_centers, &
                                lake_interface_fields,step_length)
  type(lakeparameters), pointer :: lake_parameters
  type(lakeparameters), pointer :: transposed_lake_parameters
  real(dp), pointer, dimension(:,:), intent(in) :: initial_water_to_lake_centers
  type(lakeinterfaceprognosticfields), pointer, intent(inout) :: lake_interface_fields
  real(dp), pointer, dimension(:,:)  :: transposed_initial_water_to_lake_centers
  type(lakeinterfaceprognosticfields), pointer :: transposed_lake_interface_fields
  real(dp) :: step_length
  integer, dimension(2) :: array_shape
        array_shape = shape(initial_water_to_lake_centers)
        allocate(transposed_initial_water_to_lake_centers(array_shape(2),&
                                                          array_shape(1)))
        transposed_lake_parameters => transpose_lake_parameters(lake_parameters)
        transposed_lake_interface_fields => transpose_lake_interface_fields(lake_interface_fields)
        transposed_initial_water_to_lake_centers = transpose(initial_water_to_lake_centers)
        call init_lake_model_test_orig(transposed_lake_parameters,&
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

function get_lake_prognostics() result(value)
  type(lakeprognostics), pointer :: value
        value =>  transpose_lake_prognostics(get_lake_prognostics_orig())
end function get_lake_prognostics

function get_lake_fields() result(value)
  type(lakefields), pointer :: value
        value => transpose_lake_fields(get_lake_fields_orig())
end function get_lake_fields

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

function transpose_lake_parameters(lake_parameters) result(transposed_lake_parameters)
  type(lakeparameters), pointer :: lake_parameters
  type(lakeparameters), pointer :: transposed_lake_parameters
  logical, pointer, dimension(:,:) :: lake_centers_transpose
  real(dp), pointer, dimension(:,:) :: connection_volume_thresholds_transpose
  real(dp), pointer, dimension(:,:) :: flood_volume_thresholds_transpose
  logical, pointer, dimension(:,:) :: flood_local_redirect_transpose
  logical, pointer, dimension(:,:) :: connect_local_redirect_transpose
  logical, pointer, dimension(:,:) :: additional_flood_local_redirect_transpose
  logical, pointer, dimension(:,:) :: additional_connect_local_redirect_transpose
  integer, pointer, dimension(:,:) :: merge_points_transpose
  real(dp), pointer, dimension(:,:) :: cell_areas_on_surface_model_grid_transpose
  integer, pointer, dimension(:,:) :: flood_next_cell_lon_index_transpose
  integer, pointer, dimension(:,:) :: flood_next_cell_lat_index_transpose
  integer, pointer, dimension(:,:) :: connect_next_cell_lon_index_transpose
  integer, pointer, dimension(:,:) :: connect_next_cell_lat_index_transpose
  integer, pointer, dimension(:,:) :: flood_force_merge_lon_index_transpose
  integer, pointer, dimension(:,:) :: flood_force_merge_lat_index_transpose
  integer, pointer, dimension(:,:) :: connect_force_merge_lon_index_transpose
  integer, pointer, dimension(:,:) :: connect_force_merge_lat_index_transpose
  integer, pointer, dimension(:,:) :: flood_redirect_lon_index_transpose
  integer, pointer, dimension(:,:) :: flood_redirect_lat_index_transpose
  integer, pointer, dimension(:,:) :: connect_redirect_lon_index_transpose
  integer, pointer, dimension(:,:) :: connect_redirect_lat_index_transpose
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lon_index_transpose
  integer, pointer, dimension(:,:) :: additional_flood_redirect_lat_index_transpose
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lon_index_transpose
  integer, pointer, dimension(:,:) :: additional_connect_redirect_lat_index_transpose
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lon_index_transpose
  integer, pointer, dimension(:,:) :: corresponding_surface_cell_lat_index_transpose
        allocate(lake_centers_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connection_volume_thresholds_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_volume_thresholds_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_local_redirect_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_local_redirect_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(additional_flood_local_redirect_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(additional_connect_local_redirect_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(merge_points_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(cell_areas_on_surface_model_grid_transpose(lake_parameters%nlon_surface_model,&
                                                            lake_parameters%nlat_surface_model))
        allocate(flood_next_cell_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_next_cell_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_next_cell_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_next_cell_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_force_merge_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_force_merge_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_force_merge_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_force_merge_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_redirect_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(flood_redirect_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_redirect_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(connect_redirect_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(additional_flood_redirect_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(additional_flood_redirect_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(additional_connect_redirect_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(additional_connect_redirect_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(corresponding_surface_cell_lon_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        allocate(corresponding_surface_cell_lat_index_transpose(lake_parameters%nlon,lake_parameters%nlat))
        lake_centers_transpose = transpose(lake_parameters%lake_centers)
        connection_volume_thresholds_transpose = &
          transpose(lake_parameters%connection_volume_thresholds)
        flood_volume_thresholds_transpose = &
          transpose(lake_parameters%flood_volume_thresholds)
        flood_local_redirect_transpose = &
          transpose(lake_parameters%flood_local_redirect)
        connect_local_redirect_transpose = &
          transpose(lake_parameters%connect_local_redirect)
        additional_flood_local_redirect_transpose = &
          transpose(lake_parameters%additional_flood_local_redirect)
        additional_connect_local_redirect_transpose = &
          transpose(lake_parameters%additional_connect_local_redirect)
        merge_points_transpose = transpose(lake_parameters%merge_points)
        cell_areas_on_surface_model_grid_transpose = &
          transpose(lake_parameters%cell_areas_on_surface_model_grid)
        flood_next_cell_lat_index_transpose = &
          transpose(lake_parameters%flood_next_cell_lat_index)
        flood_next_cell_lon_index_transpose = &
          transpose(lake_parameters%flood_next_cell_lon_index)
        connect_next_cell_lat_index_transpose = &
          transpose(lake_parameters%connect_next_cell_lat_index)
        connect_next_cell_lon_index_transpose = &
          transpose(lake_parameters%connect_next_cell_lon_index)
        flood_force_merge_lat_index_transpose = &
          transpose(lake_parameters%flood_force_merge_lat_index)
        flood_force_merge_lon_index_transpose = &
          transpose(lake_parameters%flood_force_merge_lon_index)
        connect_force_merge_lat_index_transpose = &
          transpose(lake_parameters%connect_force_merge_lat_index)
        connect_force_merge_lon_index_transpose = &
          transpose(lake_parameters%connect_force_merge_lon_index)
        flood_redirect_lat_index_transpose = &
          transpose(lake_parameters%flood_redirect_lat_index)
        flood_redirect_lon_index_transpose = &
          transpose(lake_parameters%flood_redirect_lon_index)
        connect_redirect_lat_index_transpose = &
          transpose(lake_parameters%connect_redirect_lat_index)
        connect_redirect_lon_index_transpose = &
          transpose(lake_parameters%connect_redirect_lon_index)
        additional_flood_redirect_lat_index_transpose = &
          transpose(lake_parameters%additional_flood_redirect_lat_index)
        additional_flood_redirect_lon_index_transpose = &
          transpose(lake_parameters%additional_flood_redirect_lon_index)
        additional_connect_redirect_lat_index_transpose = &
          transpose(lake_parameters%additional_connect_redirect_lat_index)
        additional_connect_redirect_lon_index_transpose = &
          transpose(lake_parameters%additional_connect_redirect_lon_index)
        corresponding_surface_cell_lat_index_transpose = &
          transpose(lake_parameters%corresponding_surface_cell_lat_index)
        corresponding_surface_cell_lon_index_transpose = &
          transpose(lake_parameters%corresponding_surface_cell_lon_index)
        !The initial lake parameters object will have stored the untransposed index values in the
        !arrays (lats in lons and visa versa) so here provide them reversed to correct this early
        !reversal
        transposed_lake_parameters => lakeparameters(lake_centers_transpose, &
                                                     connection_volume_thresholds_transpose, &
                                                     flood_volume_thresholds_transpose, &
                                                     flood_local_redirect_transpose, &
                                                     connect_local_redirect_transpose, &
                                                     additional_flood_local_redirect_transpose, &
                                                     additional_connect_local_redirect_transpose, &
                                                     merge_points_transpose, &
                                                     cell_areas_on_surface_model_grid_transpose, &
                                                     flood_next_cell_lat_index_transpose, &
                                                     flood_next_cell_lon_index_transpose, &
                                                     connect_next_cell_lat_index_transpose, &
                                                     connect_next_cell_lon_index_transpose, &
                                                     flood_force_merge_lat_index_transpose, &
                                                     flood_force_merge_lon_index_transpose, &
                                                     connect_force_merge_lat_index_transpose, &
                                                     connect_force_merge_lon_index_transpose, &
                                                     flood_redirect_lat_index_transpose, &
                                                     flood_redirect_lon_index_transpose, &
                                                     connect_redirect_lat_index_transpose, &
                                                     connect_redirect_lon_index_transpose, &
                                                     additional_flood_redirect_lat_index_transpose, &
                                                     additional_flood_redirect_lon_index_transpose, &
                                                     additional_connect_redirect_lat_index_transpose, &
                                                     additional_connect_redirect_lon_index_transpose, &
                                                     corresponding_surface_cell_lat_index_transpose, &
                                                     corresponding_surface_cell_lon_index_transpose, &
                                                     lake_parameters%nlon,lake_parameters%nlat, &
                                                     lake_parameters%nlon_coarse,&
                                                     lake_parameters%nlat_coarse, &
                                                     lake_parameters%nlon_surface_model,&
                                                     lake_parameters%nlat_surface_model, &
                                                     lake_parameters%instant_throughflow, &
                                                     lake_parameters%lake_retention_coefficient)
end function transpose_lake_parameters

function transpose_lake_prognostics(lake_prognostics) result(transposed_lake_prognostics)
  type(lakeprognostics), pointer :: lake_prognostics
  type(lakeprognostics), pointer :: transposed_lake_prognostics
    transposed_lake_prognostics => lake_prognostics
end function transpose_lake_prognostics

subroutine renumber_lakes(transposed_lake_fields)
  type(lakefields), pointer :: transposed_lake_fields
  integer, pointer, dimension(:,:) :: lake_numbers_transposed_relabeled
    allocate(lake_numbers_transposed_relabeled,mold=transposed_lake_fields%lake_numbers)
    lake_numbers_transposed_relabeled = transposed_lake_fields%lake_numbers
    where (transposed_lake_fields%lake_numbers == 2)
      lake_numbers_transposed_relabeled = 4
    else where (transposed_lake_fields%lake_numbers == 4)
      lake_numbers_transposed_relabeled = 5
    else where (transposed_lake_fields%lake_numbers == 5)
      lake_numbers_transposed_relabeled = 6
    else where (transposed_lake_fields%lake_numbers == 6)
      lake_numbers_transposed_relabeled = 2
    end where
    transposed_lake_fields%lake_numbers = lake_numbers_transposed_relabeled
end subroutine renumber_lakes

subroutine reorder_lake_volumes(lake_volumes)
  real(dp), pointer, dimension(:) :: lake_volumes
  real(dp), pointer, dimension(:) :: lake_volumes_orig
    allocate(lake_volumes_orig,mold=lake_volumes)
    lake_volumes_orig(:) = lake_volumes(:)
    lake_volumes(4) = lake_volumes_orig(2)
    lake_volumes(5) = lake_volumes_orig(4)
    lake_volumes(6) = lake_volumes_orig(5)
    lake_volumes(2) = lake_volumes_orig(6)
end subroutine reorder_lake_volumes

subroutine renumber_lakes_two(transposed_lake_fields)
  type(lakefields), pointer :: transposed_lake_fields
  integer, pointer, dimension(:,:) :: lake_numbers_transposed_relabeled
    allocate(lake_numbers_transposed_relabeled,mold=transposed_lake_fields%lake_numbers)
    lake_numbers_transposed_relabeled = transposed_lake_fields%lake_numbers
    where (transposed_lake_fields%lake_numbers == 4)
      lake_numbers_transposed_relabeled = 1
    else where (transposed_lake_fields%lake_numbers == 7)
      lake_numbers_transposed_relabeled = 2
    else where (transposed_lake_fields%lake_numbers == 1)
      lake_numbers_transposed_relabeled = 3
    else where (transposed_lake_fields%lake_numbers == 2)
      lake_numbers_transposed_relabeled = 4
    else where (transposed_lake_fields%lake_numbers == 3)
      lake_numbers_transposed_relabeled = 6
    else where (transposed_lake_fields%lake_numbers == 6)
      lake_numbers_transposed_relabeled = 7
    end where
    transposed_lake_fields%lake_numbers = lake_numbers_transposed_relabeled
end subroutine renumber_lakes_two

subroutine reorder_lake_volumes_two(lake_volumes)
  real(dp), pointer, dimension(:) :: lake_volumes
  real(dp), pointer, dimension(:) :: lake_volumes_orig
    allocate(lake_volumes_orig,mold=lake_volumes)
    lake_volumes_orig(:) = lake_volumes(:)
    lake_volumes(1) = lake_volumes_orig(4)
    lake_volumes(2) = lake_volumes_orig(7)
    lake_volumes(3) = lake_volumes_orig(1)
    lake_volumes(4) = lake_volumes_orig(2)
    lake_volumes(6) = lake_volumes_orig(3)
    lake_volumes(7) = lake_volumes_orig(6)
end subroutine reorder_lake_volumes_two

function transpose_lake_fields(lake_fields) result(transposed_lake_fields)
  type(lakefields), pointer :: lake_fields
  type(lakefields), pointer :: transposed_lake_fields
  integer, pointer, dimension(:) :: cells_with_lakes_lat_temp
  integer, pointer, dimension(:) :: cells_with_lakes_lon_temp
    allocate(transposed_lake_fields)
    allocate(transposed_lake_fields%connected_lake_cells,mold=lake_fields%connected_lake_cells)
    allocate(transposed_lake_fields%flooded_lake_cells,mold=lake_fields%flooded_lake_cells)
    allocate(transposed_lake_fields%lake_numbers,mold=lake_fields%lake_numbers)
    allocate(transposed_lake_fields%buried_lake_numbers,mold=lake_fields%buried_lake_numbers)
    allocate(transposed_lake_fields%effective_volume_per_cell_on_surface_grid,&
             mold=lake_fields%effective_volume_per_cell_on_surface_grid)
    allocate(transposed_lake_fields%new_effective_volume_per_cell_on_surface_grid,&
             mold=lake_fields%new_effective_volume_per_cell_on_surface_grid)
    allocate(transposed_lake_fields%effective_lake_height_on_surface_grid_from_lakes,&
             mold=lake_fields%effective_lake_height_on_surface_grid_from_lakes)
    allocate(transposed_lake_fields%effective_lake_height_on_surface_grid_to_lakes,&
             mold=lake_fields%effective_lake_height_on_surface_grid_to_lakes)
    allocate(transposed_lake_fields%evaporation_on_surface_grid,&
             mold=lake_fields%evaporation_on_surface_grid)
    allocate(transposed_lake_fields%water_to_lakes,&
             mold=lake_fields%water_to_lakes)
    allocate(transposed_lake_fields%water_to_hd,&
             mold=lake_fields%water_to_hd)
    allocate(transposed_lake_fields%lake_water_from_ocean,&
             mold=lake_fields%lake_water_from_ocean)
    allocate(transposed_lake_fields%number_lake_cells,&
             mold=lake_fields%number_lake_cells)
    allocate(cells_with_lakes_lat_temp,mold=lake_fields%cells_with_lakes_lat)
    allocate(cells_with_lakes_lon_temp,mold=lake_fields%cells_with_lakes_lon)
    cells_with_lakes_lat_temp(:) = lake_fields%cells_with_lakes_lat(:)
    cells_with_lakes_lon_temp(:) = lake_fields%cells_with_lakes_lon(:)
    transposed_lake_fields%connected_lake_cells = transpose(lake_fields%connected_lake_cells)
    transposed_lake_fields%flooded_lake_cells = transpose(lake_fields%flooded_lake_cells)
    transposed_lake_fields%lake_numbers = transpose(lake_fields%lake_numbers)
    transposed_lake_fields%buried_lake_numbers = transpose(lake_fields%buried_lake_numbers)
    transposed_lake_fields%effective_volume_per_cell_on_surface_grid = &
      transpose(lake_fields%effective_volume_per_cell_on_surface_grid)
    transposed_lake_fields%new_effective_volume_per_cell_on_surface_grid = &
      transpose(lake_fields%new_effective_volume_per_cell_on_surface_grid)
    transposed_lake_fields%effective_lake_height_on_surface_grid_from_lakes = &
      transpose(lake_fields%effective_lake_height_on_surface_grid_from_lakes)
    transposed_lake_fields%effective_lake_height_on_surface_grid_to_lakes = &
      transpose(lake_fields%effective_lake_height_on_surface_grid_to_lakes)
    transposed_lake_fields%evaporation_on_surface_grid = &
      transpose(lake_fields%evaporation_on_surface_grid)
    transposed_lake_fields%water_to_lakes = transpose(lake_fields%water_to_lakes)
    transposed_lake_fields%water_to_hd = transpose(lake_fields%water_to_hd)
    transposed_lake_fields%lake_water_from_ocean = &
      transpose(lake_fields%lake_water_from_ocean)
    transposed_lake_fields%number_lake_cells = &
      transpose(lake_fields%number_lake_cells)
    transposed_lake_fields%cells_with_lakes_lat => cells_with_lakes_lon_temp
    transposed_lake_fields%cells_with_lakes_lon => cells_with_lakes_lat_temp
end function transpose_lake_fields

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

function calculate_diagnostic_lake_volumes(lake_parameters,&
                                           lake_prognostics,&
                                           lake_fields) result(diagnostic_lake_volumes_transposed)
  type(lakeparameters), pointer, intent(in) :: lake_parameters
  type(lakeprognostics), pointer, intent(in) :: lake_prognostics
  type(lakefields), pointer, intent(in) :: lake_fields
  type(lakefields), pointer             :: lake_fields_orig
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volumes
  real(dp), dimension(:,:), pointer :: diagnostic_lake_volumes_transposed
  integer, dimension(2) :: array_shape
    lake_fields_orig => transpose_lake_fields(lake_fields)
    diagnostic_lake_volumes => calculate_diagnostic_lake_volumes_orig(lake_parameters,&
                                                                      lake_prognostics,&
                                                                      lake_fields_orig)
    array_shape = shape(diagnostic_lake_volumes)
    allocate(diagnostic_lake_volumes_transposed(array_shape(2),&
                                                array_shape(1)))
    diagnostic_lake_volumes_transposed = transpose(diagnostic_lake_volumes)
end function  calculate_diagnostic_lake_volumes

end module latlon_lake_model_interface_switcher_mod
