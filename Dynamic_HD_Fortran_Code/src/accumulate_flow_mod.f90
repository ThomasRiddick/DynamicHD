module accumulate_flow_mod
use flow_accumulation_algorithm_mod
use coords_mod
implicit none

contains

  subroutine accumulate_flow_icon_single_index(cell_neighbors, &
                                               input_river_directions, &
                                               output_cumulative_flow)
    integer, dimension(:), pointer, intent(inout) :: input_river_directions
    integer, dimension(:), pointer, intent(out) :: output_cumulative_flow
    class(*), dimension(:), pointer  :: input_river_directions_ptr
    class(*), dimension(:), pointer :: output_cumulative_flow_ptr
    integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
    integer, dimension(:,:), pointer :: secondary_neighbors
    type(icon_icosohedral_grid), pointer :: coarse_grid
    class(generic_1d_section_coords),pointer :: coarse_grid_shape
    type(icon_single_index_flow_accumulation_algorithm) :: flow_acc_alg
      where (input_river_directions ==  0 .or. &
             input_river_directions == -1 .or. &
             input_river_directions == -2 .or. &
             input_river_directions == -5)
        input_river_directions = -3
      end where
      input_river_directions_ptr => input_river_directions
      output_cumulative_flow_ptr => output_cumulative_flow
      coarse_grid => icon_icosohedral_grid(cell_neighbors)
      call coarse_grid%calculate_secondary_neighbors()
      secondary_neighbors => coarse_grid%get_cell_secondary_neighbors()
      allocate(coarse_grid_shape,source=generic_1d_section_coords(cell_neighbors, &
                                                                  secondary_neighbors))
      flow_acc_alg = &
        icon_single_index_flow_accumulation_algorithm(coarse_grid_shape, &
                                                      input_river_directions_ptr, &
                                                      output_cumulative_flow_ptr)
      call flow_acc_alg%generate_cumulative_flow(.false.)
      call flow_acc_alg%destructor()
      deallocate(coarse_grid_shape)
  end subroutine accumulate_flow_icon_single_index

end module accumulate_flow_mod
