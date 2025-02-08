module accumulate_flow_driver_mod
use accumulate_flow_mod
implicit none

contains

  subroutine accumulate_flow_latlon_f2py_wrapper(input_river_directions, &
                                                 output_cumulative_flow, &
                                                 nlat,nlon)
    integer, intent(in) :: nlat,nlon
    integer, dimension(nlat,nlon), intent(in), target :: input_river_directions
    integer, dimension(nlat,nlon), intent(out), target :: output_cumulative_flow
    integer, dimension(:,:), pointer :: input_river_directions_ptr
    integer, dimension(:,:), pointer :: output_cumulative_flow_ptr
      write(*,*) "Running flow accumulation algorithm"
      input_river_directions_ptr => input_river_directions
      output_cumulative_flow_ptr => output_cumulative_flow
      call accumulate_flow_latlon(input_river_directions_ptr, &
                                  output_cumulative_flow_ptr)
  end subroutine accumulate_flow_latlon_f2py_wrapper

  subroutine bifurcated_accumulate_flow_icon_f2py_wrapper(cell_neighbors, &
                                                          input_river_directions, &
                                                          bifurcated_next_cell_index, &
                                                          output_cumulative_flow, &
                                                          ncells)
    integer, intent(in) :: ncells
    integer, dimension(ncells,3), intent(in), target :: cell_neighbors
    integer, dimension(ncells), intent(in), target :: input_river_directions
    integer, dimension(ncells,11), intent(in), target :: bifurcated_next_cell_index
    integer, dimension(ncells), intent(out), target :: output_cumulative_flow
    integer, dimension(:,:), pointer :: cell_neighbors_ptr
    integer, dimension(:), pointer :: input_river_directions_ptr
    integer, dimension(:), pointer :: output_cumulative_flow_ptr
    integer, dimension(:,:), pointer :: bifurcated_next_cell_index_ptr
      write(*,*) "Running flow accumulation algorithm"
      cell_neighbors_ptr => cell_neighbors
      input_river_directions_ptr => input_river_directions
      output_cumulative_flow_ptr => output_cumulative_flow
      bifurcated_next_cell_index_ptr => bifurcated_next_cell_index
      call accumulate_flow_icon_single_index(cell_neighbors_ptr, &
                                             input_river_directions_ptr, &
                                             output_cumulative_flow_ptr, &
                                             bifurcated_next_cell_index_ptr)
  end subroutine bifurcated_accumulate_flow_icon_f2py_wrapper

  subroutine accumulate_flow_icon_f2py_wrapper(cell_neighbors, &
                                               input_river_directions, &
                                               output_cumulative_flow, &
                                               ncells)
    integer, intent(in) :: ncells
    integer, dimension(ncells,3), intent(in), target :: cell_neighbors
    integer, dimension(ncells), intent(in), target :: input_river_directions
    integer, dimension(ncells), intent(out), target :: output_cumulative_flow
    integer, dimension(:,:), pointer :: cell_neighbors_ptr
    integer, dimension(:), pointer :: input_river_directions_ptr
    integer, dimension(:), pointer :: output_cumulative_flow_ptr
      write(*,*) "Running flow accumulation algorithm"
      cell_neighbors_ptr => cell_neighbors
      input_river_directions_ptr => input_river_directions
      output_cumulative_flow_ptr => output_cumulative_flow
      call accumulate_flow_icon_single_index(cell_neighbors_ptr, &
                                             input_river_directions_ptr, &
                                             output_cumulative_flow_ptr)
  end subroutine accumulate_flow_icon_f2py_wrapper

end module accumulate_flow_driver_mod
