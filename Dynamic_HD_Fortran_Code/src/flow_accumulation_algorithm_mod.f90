module flow_accumulation_algorithm_mod
use subfield_mod
use coords_mod
use doubly_linked_list_mod
implicit none

type, abstract :: flow_accumulation_algorithm
  private
    class(subfield), pointer :: river_directions => null()
    class(subfield), pointer :: dependencies => null()
    class(subfield), pointer :: cumulative_flow => null()
    type(doubly_linked_list) :: q
    class(coords), pointer :: links(:)
    class(coords),pointer :: external_data_value
    class(coords),pointer :: flow_terminates_value
    class(coords),pointer :: no_data_value
    class(coords),pointer :: no_flow_value
  contains
    private
    procedure :: init_flow_accumulation_algorithm
    procedure, public :: generate_cumulative_flow
    procedure :: set_dependencies
    procedure :: add_cells_to_queue
    procedure :: process_queue
    procedure :: follow_paths
    procedure :: get_external_flow_value
    procedure :: get_flow_terminates_value
    procedure :: get_no_data_value
    procedure :: get_no_flow_value
    procedure(generate_coords_index), deferred :: generate_coords_index
    procedure(assign_coords_to_link_array), deferred :: assign_coords_to_link_array
end type flow_accumulation_algorithm

abstract interface
  function generate_coords_index(this,coords_in) result(index)
    import flow_accumulation_algorithm
    import coords
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer, intent(in) :: coords_in
    integer :: index
  end function generate_coords_index

  subroutine assign_coords_to_link_array(this,coords_index,coords_in)
    import flow_accumulation_algorithm
    import coords
    class(flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer :: coords_in
    integer :: coords_index
  end subroutine assign_coords_to_link_array
end interface

type, extends(flow_accumulation_algorithm) :: latlon_flow_accumulation_algorithm
  private
    integer :: tile_min_lat
    integer :: tile_max_lat
    integer :: tile_min_lon
    integer :: tile_max_lon
    integer :: tile_width_lat
    integer :: tile_width_lon
  contains
    procedure :: generate_coords_index => latlon_generate_coords_index
    procedure :: assign_coords_to_link_array => latlon_assign_coords_to_link_array
end type latlon_flow_accumulation_algorithm

type, extends(flow_accumulation_algorithm) :: icon_single_index_flow_accumulation_algorithm
  private

  contains
    procedure :: generate_coords_index => icon_single_index_generate_coords_index
    procedure :: assign_coords_to_link_array => icon_single_index_assign_coords_to_link_array
end type icon_single_index_flow_accumulation_algorithm

contains

subroutine init_flow_accumulation_algorithm(this)
  class(flow_accumulation_algorithm), intent(inout) :: this
    call this%dependencies%set_all(0)
    call this%cumulative_flow%set_all(0)
end subroutine init_flow_accumulation_algorithm

subroutine latlon_init_flow_accumulation_algorithm(this)
  class(latlon_flow_accumulation_algorithm), intent(inout) :: this
    allocate(this%external_data_value,source=latlon_coords(-1,-1))
    allocate(this%flow_terminates_value,source=latlon_coords(-2,-2))
    allocate(this%no_data_value,source=latlon_coords(-3,-3))
    allocate(this%no_flow_value,source=latlon_coords(-4,-4))
end subroutine latlon_init_flow_accumulation_algorithm

subroutine icon_single_index_init_flow_accumulation_algorithm(this)
  class(latlon_flow_accumulation_algorithm), intent(inout) :: this
    allocate(this%external_data_value,source=generic_1d_coords(-2))
    allocate(this%flow_terminates_value,source=generic_1d_coords(-3))
    allocate(this%no_data_value,source=generic_1d_coords(-4))
    allocate(this%no_flow_value,source=generic_1d_coords(-5))
end subroutine icon_single_index_init_flow_accumulation_algorithm

subroutine generate_cumulative_flow(this,set_links)
  class(flow_accumulation_algorithm), intent(inout) :: this
  logical, intent(in) :: set_links
  call this%river_directions%for_all(set_dependencies_wrapper,this)
  call this%dependencies%for_all(add_cells_to_queue_wrapper,this)
  call this%process_queue()
  if (set_links) call this%river_directions%for_all_edge_cells(follow_paths_wrapper,this)
end subroutine generate_cumulative_flow

subroutine set_dependencies_wrapper(this,coords_in)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(in) :: coords_in
    select type(this)
      class is (flow_accumulation_algorithm)
        call this%set_dependencies(coords_in)
    end select
end subroutine set_dependencies_wrapper

subroutine set_dependencies(this,coords_in)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(in) :: coords_in
  class(coords),pointer :: target_coords
  class(coords),pointer :: target_of_target_coords
  class(*), pointer      :: target_coords_ptr
  class(*), pointer      :: target_of_target_coords_ptr
  class(*), pointer  :: dependency_ptr
    target_coords_ptr => this%river_directions%get_value(coords_in)
    select type(target_coords_ptr)
        class is (coords)
          target_coords => target_coords_ptr
    end select
    select type(target_of_target_coords_ptr)
        class is (coords)
          target_of_target_coords => target_of_target_coords_ptr
    end select
    target_of_target_coords_ptr => this%river_directions%get_value(target_coords)
    if ( target_coords%are_equal_to(this%get_no_data_value()) ) then
      call this%cumulative_flow%set_value(coords_in,this%get_no_data_value())
    else if ( .not. (target_coords%are_equal_to(this%get_no_flow_value()) .or. &
                     target_of_target_coords%are_equal_to(this%get_no_flow_value()) ) ) then
      dependency_ptr => this%dependencies%get_value(target_coords)
      select type (dependency_ptr)
        type is (integer)
          call this%dependencies%set_value(target_coords, &
                                           dependency_ptr+1)
      end select
    end if
end subroutine set_dependencies

subroutine add_cells_to_queue_wrapper(this,coords_in)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(in) :: coords_in
    select type(this)
    class is (flow_accumulation_algorithm)
      call this%add_cells_to_queue(coords_in)
    end select
end subroutine add_cells_to_queue_wrapper

subroutine add_cells_to_queue(this,coords_in)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(in) :: coords_in
  class(coords), pointer :: target_coords
  class(*), pointer  :: target_coords_ptr
  class(*), pointer  :: dependency_ptr
  target_coords_ptr => this%river_directions%get_value(coords_in)
  select type(target_coords_ptr)
      class is (coords)
        target_coords => target_coords_ptr
  end select
  dependency_ptr => this%dependencies%get_value(coords_in)
  select type (dependency_ptr)
    type is (integer)
    if (  dependency_ptr == 0 .and. &
         ( .not. target_coords%are_equal_to(this%get_no_data_value()) ) ) then
      call this%q%add_value_to_back(coords_in)
    end if
  end select
end subroutine add_cells_to_queue

subroutine process_queue(this)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer :: current_coords
  class(*),      pointer :: current_coords_ptr
  class(coords), pointer :: target_coords
  class(*), pointer      :: target_coords_ptr
  class(coords), pointer :: target_of_target_coords
  class(*), pointer      :: target_of_target_coords_ptr
  class(*), pointer :: cumulative_flow_current_coords_ptr
  class(*), pointer :: cumulative_flow_target_coords_ptr
  class(*), pointer :: dependency_ptr
  integer :: dependency
  if ( this%q%iterate_forward() ) stop 'Edge cell list is not correctly initialised'
  do while ( this%q%get_length() > 0 )
    current_coords_ptr => this%q%get_value_at_iterator_position()
    select type(current_coords_ptr)
      class is (coords)
        current_coords => current_coords_ptr
    end select
    call this%q%remove_element_at_iterator_position()
    target_coords_ptr => this%river_directions%get_value(current_coords)
    select type(target_coords_ptr)
      class is (coords)
        target_coords => target_coords_ptr
    end select
    target_of_target_coords_ptr => this%river_directions%get_value(target_coords)
    select type(target_of_target_coords_ptr)
      class is (coords)
        target_of_target_coords => target_of_target_coords_ptr
    end select
    if ( target_coords%are_equal_to(this%get_no_data_value()) .or. &
         target_coords%are_equal_to(this%get_no_flow_value()) .or. &
         target_of_target_coords%are_equal_to(this%get_no_data_value()) ) then
      cycle
    end if
    cumulative_flow_target_coords_ptr => this%cumulative_flow%get_value(target_coords)
    cumulative_flow_current_coords_ptr => this%cumulative_flow%get_value(current_coords)
    select type (cumulative_flow_target_coords_ptr)
      type is (integer)
        select type(cumulative_flow_current_coords_ptr)
          type is (integer)
            call this%cumulative_flow%set_value(target_coords, &
                                                cumulative_flow_target_coords_ptr + &
                                                cumulative_flow_current_coords_ptr)
        end select
      end select
    dependency_ptr => this%dependencies%get_value(target_coords)
    select type (dependency_ptr)
      type is (integer)
      dependency = dependency_ptr - 1
    end select
    call this%dependencies%set_value(target_coords, &
                                     dependency)
    if ( dependency == 0 ) then
      call this%q%add_value_to_back(target_coords)
    end if
  end do
end subroutine process_queue

subroutine follow_paths_wrapper(this,initial_coords)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(in) :: initial_coords
    select type(this)
    class is (flow_accumulation_algorithm)
      call this%follow_paths(initial_coords)
    end select
end subroutine follow_paths_wrapper

subroutine follow_paths(this,initial_coords)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(in) :: initial_coords
  class(coords), pointer :: current_coords
  class(*), pointer :: target_coords_ptr
  class(coords), pointer :: target_coords
  integer :: coords_index
  allocate(current_coords,source=initial_coords)
  coords_index = this%generate_coords_index(initial_coords)
  do
    if ( target_coords%are_equal_to(this%get_no_data_value()) .or. &
         target_coords%are_equal_to(this%get_no_flow_value()) ) then
      call this%assign_coords_to_link_array(coords_index,this%get_flow_terminates_value())
      exit
    end if
    target_coords_ptr => this%river_directions%get_value(current_coords)
    select type (target_coords_ptr)
      class is (coords)
        target_coords => target_coords_ptr
    end select
    if ( this%river_directions%coords_outside_subfield(target_coords) ) then
      if ( current_coords%are_equal_to(initial_coords) ) then
         call this%assign_coords_to_link_array(coords_index,this%get_external_flow_value())
      else
         call this%assign_coords_to_link_array(coords_index,current_coords)
      end if
      exit
    end if
    current_coords => target_coords
  end do
end subroutine follow_paths

  function get_external_flow_value(this) result(external_data_value)
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: external_data_value
      external_data_value => this%external_data_value
  end function get_external_flow_value

  function get_flow_terminates_value(this) result(flow_terminates_value)
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: flow_terminates_value
      flow_terminates_value => this%flow_terminates_value
  end function get_flow_terminates_value

  function get_no_data_value(this) result(no_data_value)
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer ::no_data_value
      no_data_value => this%no_data_value
  end function get_no_data_value

  function get_no_flow_value(this) result(no_flow_value)
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer ::no_flow_value
      no_flow_value => this%no_flow_value
  end function get_no_flow_value

  function latlon_generate_coords_index(this,coords_in) result(index)
    class(latlon_flow_accumulation_algorithm), intent (in) :: this
    class(coords), intent (in) :: coords_in
    integer :: index
    integer :: relative_lat, relative_lon
    select type (coords_in)
    class is (latlon_coords)
      relative_lat = 1 + coords_in%lat - this%tile_min_lat
      relative_lon = 1 + coords_in%lon - this%tile_min_lon
    end select
    if ( relative_lat == 1 ) then
      index = relative_lon
    else if ( relative_lat < this%tile_width_lat .and. &
              relative_lat > 1 ) then
      if ( relative_lon == 1 ) then
        index = 2*this%tile_width_lon + 2*this%tile_width_lat - relative_lat - 2
      else if ( relative_lon == this%tile_width_lon ) then
        index = this%tile_width_lon + relative_lat - 1
      else
        stop 'trying to generate the index of an non edge cell'
      end if
    else if ( relative_lat == this%tile_width_lat ) then
      index = this%tile_width_lat + 2*this%tile_width_lon - relative_lon - 1
    else
        stop 'trying to generate the index of an non edge cell'
    end if
  end function latlon_generate_coords_index


  subroutine latlon_assign_coords_to_link_array(this,coords_index,coords_in)
    class(latlon_flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer :: coords_in
    integer :: coords_index
    select type(latlon_links => this%links)
      type is (latlon_coords)
      select type(coords_in)
        type is (latlon_coords)
          latlon_links(coords_index) = coords_in
      end select
    end select
  end subroutine latlon_assign_coords_to_link_array

  function icon_single_index_generate_coords_index(this,coords_in) result(index)
    class(icon_single_index_flow_accumulation_algorithm), intent (in) :: this
    class(coords), intent (in) :: coords_in
    integer :: index
      ! Dummy code to prevent compiler warnings
      select type (coords_in)
      type is (latlon_coords)
        index = coords_in%lat
      end select
      ! Dummy code to prevent compiler warnings
      select type (ndv=> this%no_data_value)
      type is (latlon_coords)
      index = ndv%lat
      end select
      index = 1
      stop "Function generate_coords_index not yet implemented for incon grid"
  end function icon_single_index_generate_coords_index

  subroutine icon_single_index_assign_coords_to_link_array(this,coords_index,coords_in)
    class(icon_single_index_flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer :: coords_in
    integer :: coords_index
    select type(generic_1d_links => this%links)
      type is (generic_1d_coords)
      select type(coords_in)
        type is (generic_1d_coords)
          generic_1d_links(coords_index) = coords_in
      end select
    end select
  end subroutine icon_single_index_assign_coords_to_link_array

end module flow_accumulation_algorithm_mod
