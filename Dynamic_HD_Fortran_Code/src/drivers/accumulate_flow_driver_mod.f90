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

end module accumulate_flow_driver_mod
