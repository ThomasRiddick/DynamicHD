module accumulate_flow_mod
use flow_accumulation_algorithm_mod
use coords_mod
implicit none

contains

  subroutine accumulate_flow_icon_single_index(cell_neighbors, &
                                               input_river_directions, &
                                               output_cumulative_flow, &
                                               bifurcated_next_cell_index)
    integer, dimension(:), pointer, intent(inout) :: input_river_directions
    integer, dimension(:), pointer, intent(out) :: output_cumulative_flow
    integer, dimension(:,:), pointer, intent(in), optional :: bifurcated_next_cell_index
    class(*), dimension(:), pointer  :: input_river_directions_ptr
    class(*), dimension(:,:), pointer  :: input_bifurcated_river_directions_ptr
    class(*), dimension(:), pointer :: output_cumulative_flow_ptr
    integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
    integer, dimension(:,:), pointer :: secondary_neighbors
    type(icon_icosohedral_grid), pointer :: coarse_grid
    type(generic_1d_section_coords) :: coarse_grid_shape
    type(icon_single_index_flow_accumulation_algorithm) :: flow_acc_alg
      where (input_river_directions ==  0 .or. &
             input_river_directions == -1 .or. &
             input_river_directions == -2 .or. &
             input_river_directions == -5)
        input_river_directions = -3
      end where
      input_river_directions_ptr => input_river_directions
      output_cumulative_flow_ptr => output_cumulative_flow
      if (present(bifurcated_next_cell_index)) then
        input_bifurcated_river_directions_ptr => bifurcated_next_cell_index
      end if
      coarse_grid => icon_icosohedral_grid(cell_neighbors)
      call coarse_grid%calculate_secondary_neighbors()
      secondary_neighbors => coarse_grid%get_cell_secondary_neighbors()
      coarse_grid_shape = generic_1d_section_coords(cell_neighbors, &
                                                    secondary_neighbors)
      if (present(bifurcated_next_cell_index)) then
        flow_acc_alg = &
          icon_single_index_flow_accumulation_algorithm(coarse_grid_shape, &
                                                        input_river_directions_ptr, &
                                                        output_cumulative_flow_ptr, &
                                                        input_bifurcated_river_directions_ptr)
      else
        flow_acc_alg = &
          icon_single_index_flow_accumulation_algorithm(coarse_grid_shape, &
                                                        input_river_directions_ptr, &
                                                        output_cumulative_flow_ptr)
      end if
      call flow_acc_alg%generate_cumulative_flow(.false.)
      if (present(bifurcated_next_cell_index)) then
        call flow_acc_alg%update_bifurcated_flows()
      end if
      call flow_acc_alg%icon_single_index_destructor()
      call coarse_grid_shape%generic_1d_section_coords_destructor()
      deallocate(coarse_grid)
      deallocate(secondary_neighbors)
  end subroutine accumulate_flow_icon_single_index

end module accumulate_flow_mod
