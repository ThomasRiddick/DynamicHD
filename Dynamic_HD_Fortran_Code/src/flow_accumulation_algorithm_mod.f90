module flow_accumulation_algorithm_mod
use subfield_mod
use coords_mod
use doubly_linked_list_mod
implicit none

type, abstract :: flow_accumulation_algorithm
  private
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
    procedure, public :: destructor
    procedure(get_next_cell_coords), deferred :: get_next_cell_coords
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


  function get_next_cell_coords(this,coords_in) result(next_cell_coords)
    import flow_accumulation_algorithm
    import coords
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: coords_in
    class(coords), pointer :: next_cell_coords
  end function get_next_cell_coords
end interface

type, extends(flow_accumulation_algorithm) :: latlon_flow_accumulation_algorithm
  private
    integer :: tile_min_lat
    integer :: tile_max_lat
    integer :: tile_min_lon
    integer :: tile_max_lon
    integer :: tile_width_lat
    integer :: tile_width_lon
    class(subfield), pointer :: next_cell_index_lat => null()
    class(subfield), pointer :: next_cell_index_lon => null()
    class(subfield), pointer :: river_directions => null()
  contains
    procedure :: generate_coords_index => latlon_generate_coords_index
    procedure :: assign_coords_to_link_array => latlon_assign_coords_to_link_array
    procedure :: get_next_cell_coords => latlon_get_next_cell_coords
end type latlon_flow_accumulation_algorithm

type, extends(flow_accumulation_algorithm) :: icon_single_index_flow_accumulation_algorithm
  private
    class(subfield), pointer :: next_cell_index => null()
  contains
    procedure :: icon_single_index_init_flow_accumulation_algorithm
    procedure :: icon_single_index_destructor
    procedure :: generate_coords_index => icon_single_index_generate_coords_index
    procedure :: assign_coords_to_link_array => icon_single_index_assign_coords_to_link_array
    procedure :: get_next_cell_coords => icon_single_index_get_next_cell_coords
end type icon_single_index_flow_accumulation_algorithm

interface icon_single_index_flow_accumulation_algorithm
  procedure :: icon_single_index_flow_accumulation_algorithm_constructor
end interface icon_single_index_flow_accumulation_algorithm

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

subroutine icon_single_index_init_flow_accumulation_algorithm(this,field_section_coords, &
                                                              next_cell_index, &
                                                              cumulative_flow)
  class(icon_single_index_flow_accumulation_algorithm), intent(inout) :: this
  class(*), dimension(:), pointer, intent(in) :: next_cell_index
  class(*), dimension(:), pointer, intent(out) :: cumulative_flow
  type(generic_1d_section_coords), intent(in) :: field_section_coords
  class(*), dimension(:), pointer :: dependencies_data
    this%next_cell_index => icon_single_index_subfield(next_cell_index,field_section_coords)
    this%cumulative_flow => icon_single_index_subfield(cumulative_flow,field_section_coords)
    allocate(integer::dependencies_data(size(next_cell_index)))
    this%dependencies => icon_single_index_subfield(dependencies_data, &
                                                    field_section_coords)
    allocate(this%external_data_value,source=generic_1d_coords(-2,.true.))
    allocate(this%flow_terminates_value,source=generic_1d_coords(-3,.true.))
    allocate(this%no_data_value,source=generic_1d_coords(-4,.true.))
    allocate(this%no_flow_value,source=generic_1d_coords(-5,.true.))
    call this%init_flow_accumulation_algorithm()
end subroutine icon_single_index_init_flow_accumulation_algorithm

function icon_single_index_flow_accumulation_algorithm_constructor(field_section_coords, &
                                                                   next_cell_index, &
                                                                   cumulative_flow) &
                                                                   result(constructor)
  class(*), dimension(:), pointer, intent(in) :: next_cell_index
  class(*), dimension(:), pointer, intent(out) :: cumulative_flow
  type(generic_1d_section_coords), intent(in) :: field_section_coords
  type(icon_single_index_flow_accumulation_algorithm) :: constructor
    call constructor%icon_single_index_init_flow_accumulation_algorithm(field_section_coords, &
                                                                        next_cell_index, &
                                                                        cumulative_flow)
end function icon_single_index_flow_accumulation_algorithm_constructor

subroutine destructor(this)
  class(flow_accumulation_algorithm), intent(inout) :: this
    call this%dependencies%destructor()
    deallocate(this%dependencies)
    deallocate(this%cumulative_flow)
    deallocate(this%external_data_value)
    deallocate(this%flow_terminates_value)
    deallocate(this%no_data_value)
    deallocate(this%no_flow_value)
    call this%q%destructor()
end subroutine destructor

subroutine icon_single_index_destructor(this)
  class(icon_single_index_flow_accumulation_algorithm), intent(inout) :: this
    deallocate(this%next_cell_index)
    call this%destructor
end subroutine

subroutine generate_cumulative_flow(this,set_links)
  class(flow_accumulation_algorithm), intent(inout) :: this
  logical, intent(in) :: set_links
    call this%dependencies%for_all(set_dependencies_wrapper,this)
    call this%dependencies%for_all(add_cells_to_queue_wrapper,this)
    call this%process_queue()
    if (set_links) call this%dependencies%for_all_edge_cells(follow_paths_wrapper,this)
end subroutine generate_cumulative_flow

subroutine set_dependencies_wrapper(this,coords_in)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(inout) :: coords_in
    select type(this)
      class is (flow_accumulation_algorithm)
        call this%set_dependencies(coords_in)
    end select
end subroutine set_dependencies_wrapper

subroutine set_dependencies(this,coords_in)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(inout) :: coords_in
  class(coords),pointer :: target_coords
  class(coords),pointer :: target_of_target_coords
  class(*), pointer  :: dependency_ptr
    if ( .not. (coords_in%are_equal_to(this%get_no_data_value()) .or. &
                coords_in%are_equal_to(this%get_no_flow_value()) .or. &
                coords_in%are_equal_to(this%get_flow_terminates_value()) )) then
      target_coords => this%get_next_cell_coords(coords_in)
      if ( target_coords%are_equal_to(this%get_no_data_value()) ) then
        call this%cumulative_flow%set_value(coords_in,this%get_no_data_value())
      else if ( target_coords%are_equal_to(this%get_no_flow_value())) then
        call this%cumulative_flow%set_value(coords_in,this%get_no_flow_value())
      else if ( .not. target_coords%are_equal_to(this%get_flow_terminates_value())) then
        target_of_target_coords => this%get_next_cell_coords(target_coords)
        if ( .not. (target_of_target_coords%are_equal_to(this%get_no_flow_value()) .or. &
                    target_of_target_coords%are_equal_to(this%get_no_data_value()) )) then
          dependency_ptr => this%dependencies%get_value(target_coords)
          select type (dependency_ptr)
            type is (integer)
              call this%dependencies%set_value(target_coords, &
                                               dependency_ptr+1)
          end select
          deallocate(dependency_ptr)
        end if
        deallocate(target_of_target_coords)
      end if
      deallocate(target_coords)
    end if
    deallocate(coords_in)
end subroutine set_dependencies

subroutine add_cells_to_queue_wrapper(this,coords_in)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(inout) :: coords_in
    select type(this)
    class is (flow_accumulation_algorithm)
      call this%add_cells_to_queue(coords_in)
    end select
end subroutine add_cells_to_queue_wrapper

subroutine add_cells_to_queue(this,coords_in)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(inout) :: coords_in
  class(coords), pointer :: target_coords
  class(*), pointer  :: dependency_ptr
  target_coords => this%get_next_cell_coords(coords_in)
  dependency_ptr => this%dependencies%get_value(coords_in)
  select type (dependency_ptr)
    type is (integer)
    if (  dependency_ptr == 0 .and. &
         ( .not. target_coords%are_equal_to(this%get_no_data_value()) ) ) then
      call this%q%add_value_to_back(coords_in)
      if ( .not. target_coords%are_equal_to(this%get_flow_terminates_value())) then
        call this%cumulative_flow%set_value(coords_in,1)
      end if
    end if
  end select
  deallocate(coords_in)
  deallocate(dependency_ptr)
  deallocate(target_coords)
end subroutine add_cells_to_queue

subroutine process_queue(this)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer :: current_coords
  class(*),      pointer :: current_coords_ptr
  class(coords), pointer :: target_coords
  class(coords), pointer :: target_of_target_coords
  class(*), pointer :: cumulative_flow_current_coords_ptr
  class(*), pointer :: cumulative_flow_target_coords_ptr
  class(*), pointer :: dependency_ptr
  integer :: dependency
  do while ( .not. this%q%iterate_forward() )
    current_coords_ptr => this%q%get_value_at_iterator_position()
    select type(current_coords_ptr)
      class is (coords)
        current_coords => current_coords_ptr
    end select
    target_coords => this%get_next_cell_coords(current_coords)
    if ( target_coords%are_equal_to(this%get_no_data_value()) .or. &
         target_coords%are_equal_to(this%get_no_flow_value()) .or. &
         target_coords%are_equal_to(this%get_flow_terminates_value())) then
      call this%q%remove_element_at_iterator_position()
      deallocate(target_coords)
      cycle
    end if
    target_of_target_coords => this%get_next_cell_coords(target_coords)
    if ( target_of_target_coords%are_equal_to(this%get_no_data_value())) then
      call this%q%remove_element_at_iterator_position()
      deallocate(target_of_target_coords)
      cycle
    end if
    deallocate(target_of_target_coords)
    dependency_ptr => this%dependencies%get_value(target_coords)
    select type (dependency_ptr)
      type is (integer)
      dependency = dependency_ptr - 1
    end select
    deallocate(dependency_ptr)
    call this%dependencies%set_value(target_coords, &
                                     dependency)
    cumulative_flow_target_coords_ptr => this%cumulative_flow%get_value(target_coords)
    cumulative_flow_current_coords_ptr => this%cumulative_flow%get_value(current_coords)
    call this%q%remove_element_at_iterator_position()
    select type (cumulative_flow_target_coords_ptr)
      type is (integer)
        select type(cumulative_flow_current_coords_ptr)
          type is (integer)
            if (dependency == 0 .and. .not. &
                target_coords%are_equal_to(this%get_flow_terminates_value())) then
              call this%cumulative_flow%set_value(target_coords, &
                                                  cumulative_flow_target_coords_ptr + &
                                                  cumulative_flow_current_coords_ptr + 1)
              call this%q%add_value_to_back(target_coords)
            else
              call this%cumulative_flow%set_value(target_coords, &
                                                  cumulative_flow_target_coords_ptr + &
                                                  cumulative_flow_current_coords_ptr)
            end if
        end select
      end select
      deallocate(target_coords)
      deallocate(cumulative_flow_target_coords_ptr)
      deallocate(cumulative_flow_current_coords_ptr)
  end do
end subroutine process_queue

subroutine follow_paths_wrapper(this,initial_coords)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(inout) :: initial_coords
    select type(this)
    class is (flow_accumulation_algorithm)
      call this%follow_paths(initial_coords)
    end select
end subroutine follow_paths_wrapper

subroutine follow_paths(this,initial_coords)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(inout) :: initial_coords
  class(coords), pointer :: current_coords
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
    target_coords => this%get_next_cell_coords(current_coords)
    if ( this%dependencies%coords_outside_subfield(target_coords) ) then
      if ( current_coords%are_equal_to(initial_coords) ) then
         call this%assign_coords_to_link_array(coords_index,this%get_external_flow_value())
      else
         call this%assign_coords_to_link_array(coords_index,current_coords)
      end if
      exit
    end if
    current_coords => target_coords
  end do
  deallocate(initial_coords)
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

  function latlon_get_next_cell_coords(this,coords_in) result(next_cell_coords)
    class(latlon_flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: coords_in
    class(coords), pointer :: next_cell_coords
    class(*), pointer :: next_cell_coords_lat_ptr
    class(*), pointer :: next_cell_coords_lon_ptr
      next_cell_coords_lat_ptr => this%next_cell_index_lat%get_value(coords_in)
      next_cell_coords_lon_ptr => this%next_cell_index_lon%get_value(coords_in)
      select type (next_cell_coords_lat_ptr)
      type is (integer)
        select type (next_cell_coords_lon_ptr)
        type is (integer)
          allocate(next_cell_coords, &
                   source=latlon_coords(next_cell_coords_lat_ptr, &
                                        next_cell_coords_lon_ptr))
        end select
      end select
  end function latlon_get_next_cell_coords

  function latlon_generate_coords_index(this,coords_in) result(index)
    class(latlon_flow_accumulation_algorithm), intent (in) :: this
    class(coords), pointer, intent (in) :: coords_in
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

  function icon_single_index_get_next_cell_coords(this,coords_in) result(next_cell_coords)
    class(icon_single_index_flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: coords_in
    class(coords), pointer :: next_cell_coords
    class(*), pointer :: next_cell_coords_index_ptr
      next_cell_coords_index_ptr => this%next_cell_index%get_value(coords_in)
      select type (next_cell_coords_index_ptr)
        type is (integer)
          allocate(next_cell_coords, &
                   source=generic_1d_coords(next_cell_coords_index_ptr,.true.))
      end select
      deallocate(next_cell_coords_index_ptr)
  end function icon_single_index_get_next_cell_coords

  function icon_single_index_generate_coords_index(this,coords_in) result(index)
    class(icon_single_index_flow_accumulation_algorithm), intent (in) :: this
    class(coords), pointer, intent (in) :: coords_in
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
