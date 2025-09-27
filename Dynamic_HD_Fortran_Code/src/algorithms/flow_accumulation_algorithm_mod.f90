module flow_accumulation_algorithm_mod
use subfield_mod
use coords_mod
use doubly_linked_list_mod
implicit none

type :: subfield_ptr
  class(subfield), pointer :: ptr
end type

type, abstract :: flow_accumulation_algorithm
  private
    class(subfield), pointer :: dependencies => null()
    class(subfield), pointer :: cumulative_flow => null()
    class(subfield_ptr), pointer, dimension(:) :: bifurcation_complete => null()
    type(doubly_linked_list) :: q
    class(coords), pointer :: links(:)
    class(coords),pointer :: external_data_value
    class(coords),pointer :: flow_terminates_value
    class(coords),pointer :: no_data_value
    class(coords),pointer :: no_flow_value
    integer :: max_neighbors
    integer :: no_bifurcation_value
    logical :: search_for_loops = .true.
  contains
    private
    procedure :: init_flow_accumulation_algorithm
    procedure, public :: generate_cumulative_flow
    procedure, public :: update_bifurcated_flows
    procedure :: set_dependencies
    procedure :: add_cells_to_queue
    procedure :: process_queue
    procedure :: follow_paths
    procedure :: get_external_flow_value
    procedure :: get_flow_terminates_value
    procedure :: get_no_data_value
    procedure :: get_no_flow_value
    procedure :: check_for_bifurcations_in_cell
    procedure :: update_bifurcated_flow
    procedure :: label_loop
    procedure, public :: destructor
    procedure(get_next_cell_coords), deferred :: get_next_cell_coords
    procedure(generate_coords_index), deferred :: generate_coords_index
    procedure(assign_coords_to_link_array), deferred :: assign_coords_to_link_array
    procedure(is_bifurcated), deferred :: is_bifurcated
    procedure(get_next_cell_bifurcated_coords), deferred :: get_next_cell_bifurcated_coords
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

  function get_next_cell_bifurcated_coords(this,coords_in,layer_in) &
    result(next_cell_coords)
    import flow_accumulation_algorithm
    import coords
    class(flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: coords_in
    class(coords), pointer :: next_cell_coords
    integer, intent(in) :: layer_in
  end function get_next_cell_bifurcated_coords

  function is_bifurcated(this,coords_in,layer_in) result(bifurcated)
    import flow_accumulation_algorithm
    import coords
    class(flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer :: coords_in
    integer, optional, intent(in) :: layer_in
    logical :: bifurcated
  end function is_bifurcated
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
    class(subfield_ptr), pointer, dimension(:) :: bifurcated_next_cell_index_lat => null()
    class(subfield_ptr), pointer, dimension(:) :: bifurcated_next_cell_index_lon => null()
    class(subfield), pointer :: river_directions => null()
  contains
    procedure :: latlon_init_flow_accumulation_algorithm
    procedure :: latlon_destructor
    procedure :: generate_coords_index => latlon_generate_coords_index
    procedure :: assign_coords_to_link_array => latlon_assign_coords_to_link_array
    procedure :: get_next_cell_coords => latlon_get_next_cell_coords
    procedure :: is_bifurcated => latlon_is_bifurcated
    procedure :: get_next_cell_bifurcated_coords => latlon_get_next_cell_bifurcated_coords
end type latlon_flow_accumulation_algorithm

interface latlon_flow_accumulation_algorithm
  procedure :: latlon_flow_accumulation_algorithm_constructor
end interface latlon_flow_accumulation_algorithm

type, extends(flow_accumulation_algorithm) :: icon_single_index_flow_accumulation_algorithm
  private
    class(subfield), pointer :: next_cell_index => null()
    class(subfield_ptr), pointer, dimension(:) :: bifurcated_next_cell_index => null()
  contains
    procedure :: icon_single_index_init_flow_accumulation_algorithm
    procedure :: icon_single_index_destructor
    procedure :: generate_coords_index => icon_single_index_generate_coords_index
    procedure :: assign_coords_to_link_array => icon_single_index_assign_coords_to_link_array
    procedure :: get_next_cell_coords => icon_single_index_get_next_cell_coords
    procedure :: is_bifurcated => icon_single_index_is_bifurcated
    procedure :: get_next_cell_bifurcated_coords => icon_single_index_get_next_cell_bifurcated_coords
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

subroutine latlon_init_flow_accumulation_algorithm(this, &
                                                   next_cell_index_lat, &
                                                   next_cell_index_lon, &
                                                   cumulative_flow)
  class(latlon_flow_accumulation_algorithm), intent(inout) :: this
  class(*), dimension(:,:), pointer, intent(in) :: next_cell_index_lat
  class(*), dimension(:,:), pointer, intent(in) :: next_cell_index_lon
  class(*), dimension(:,:), pointer, intent(inout) :: cumulative_flow
  class(*), dimension(:,:), pointer :: dependencies_data
  type(latlon_section_coords), allocatable :: field_section_coords
    allocate(field_section_coords)
    field_section_coords = latlon_section_coords(1,1,size(next_cell_index_lat,1), &
                                                     size(next_cell_index_lat,2))
    this%next_cell_index_lat => latlon_subfield(next_cell_index_lat,field_section_coords,.true.)
    this%next_cell_index_lon => latlon_subfield(next_cell_index_lon,field_section_coords,.true.)
    this%cumulative_flow => latlon_subfield(cumulative_flow,field_section_coords,.true.)
    allocate(integer::dependencies_data(size(next_cell_index_lat,1),&
                                        size(next_cell_index_lon,2)))
    this%dependencies => latlon_subfield(dependencies_data, &
                                         field_section_coords,.true.)
    allocate(this%external_data_value,source=latlon_coords(-1,-1))
    allocate(this%flow_terminates_value,source=latlon_coords(-2,-2))
    allocate(this%no_data_value,source=latlon_coords(-3,-3))
    allocate(this%no_flow_value,source=latlon_coords(-4,-4))
    call this%init_flow_accumulation_algorithm()
end subroutine latlon_init_flow_accumulation_algorithm

function latlon_flow_accumulation_algorithm_constructor(next_cell_index_lat, &
                                                        next_cell_index_lon, &
                                                        cumulative_flow) &
                                                        result(constructor)
  class(*), dimension(:,:), pointer, intent(in) :: next_cell_index_lat
  class(*), dimension(:,:), pointer, intent(in) :: next_cell_index_lon
  class(*), dimension(:,:), pointer, intent(inout) :: cumulative_flow
  type(latlon_flow_accumulation_algorithm) :: constructor
    call constructor%latlon_init_flow_accumulation_algorithm(next_cell_index_lat, &
                                                             next_cell_index_lon, &
                                                             cumulative_flow)
end function latlon_flow_accumulation_algorithm_constructor

subroutine icon_single_index_init_flow_accumulation_algorithm(this,field_section_coords, &
                                                              next_cell_index, &
                                                              cumulative_flow, &
                                                              bifurcated_next_cell_index)
  class(icon_single_index_flow_accumulation_algorithm), intent(inout) :: this
  class(*), dimension(:), pointer, intent(in) :: next_cell_index
  class(*), dimension(:), pointer, intent(out) :: cumulative_flow
  class(*), dimension(:,:), pointer, intent(in), optional :: bifurcated_next_cell_index
  class(*), dimension(:), pointer :: bifurcated_next_cell_index_slice
  class(*), dimension(:), pointer :: bifurcation_complete_slice
   type(generic_1d_section_coords), intent(in) :: field_section_coords
  class(*), dimension(:), pointer :: dependencies_data
  integer :: i
    this%max_neighbors = 12
    this%no_bifurcation_value = -9
    this%next_cell_index => icon_single_index_subfield(next_cell_index,field_section_coords)
    this%cumulative_flow => icon_single_index_subfield(cumulative_flow,field_section_coords)
    allocate(integer::dependencies_data(size(next_cell_index)))
    this%dependencies => icon_single_index_subfield(dependencies_data, &
                                                    field_section_coords)
    allocate(this%external_data_value,source=generic_1d_coords(-2,.true.))
    allocate(this%flow_terminates_value,source=generic_1d_coords(-3,.true.))
    allocate(this%no_data_value,source=generic_1d_coords(-4,.true.))
    allocate(this%no_flow_value,source=generic_1d_coords(-5,.true.))
    if (present(bifurcated_next_cell_index)) then
      allocate(subfield_ptr::this%bifurcated_next_cell_index(this%max_neighbors-1))
      allocate(subfield_ptr::this%bifurcation_complete(this%max_neighbors-1))
      do i = 1,this%max_neighbors - 1
        bifurcated_next_cell_index_slice => bifurcated_next_cell_index(:,i)
        this%bifurcated_next_cell_index(i)%ptr => &
          icon_single_index_subfield(bifurcated_next_cell_index_slice,field_section_coords)
        allocate(logical::bifurcation_complete_slice(size(next_cell_index)))
        select type(bifurcation_complete_slice)
          type is (logical)
            bifurcation_complete_slice(:) = .false.
        end select
        this%bifurcation_complete(i)%ptr => &
          icon_single_index_subfield(bifurcation_complete_slice,field_section_coords)
      end do
    end if
    call this%init_flow_accumulation_algorithm()
end subroutine icon_single_index_init_flow_accumulation_algorithm

function icon_single_index_flow_accumulation_algorithm_constructor(field_section_coords, &
                                                                   next_cell_index, &
                                                                   cumulative_flow, &
                                                                   bifurcated_next_cell_index) &
                                                                   result(constructor)
  class(*), dimension(:), pointer, intent(in) :: next_cell_index
  class(*), dimension(:), pointer, intent(out) :: cumulative_flow
  type(generic_1d_section_coords), intent(in) :: field_section_coords
  class(*), dimension(:,:), pointer, intent(in), optional :: bifurcated_next_cell_index
  type(icon_single_index_flow_accumulation_algorithm) :: constructor
    if(present(bifurcated_next_cell_index)) then
      call constructor%icon_single_index_init_flow_accumulation_algorithm(field_section_coords, &
                                                                          next_cell_index, &
                                                                          cumulative_flow, &
                                                                          bifurcated_next_cell_index)
    else
      call constructor%icon_single_index_init_flow_accumulation_algorithm(field_section_coords, &
                                                                          next_cell_index, &
                                                                          cumulative_flow)
    end if
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

subroutine latlon_destructor(this)
  class(latlon_flow_accumulation_algorithm), intent(inout) :: this
    deallocate(this%next_cell_index_lat)
    deallocate(this%next_cell_index_lon)
    call this%destructor
end subroutine latlon_destructor

subroutine icon_single_index_destructor(this)
  class(icon_single_index_flow_accumulation_algorithm), intent(inout) :: this
    deallocate(this%next_cell_index)
    !Not deallocating causes a leak but deallocating causes a seg-fault
    !futher investigation required
    !if (associated(this%bifurcation_complete)) then
    !  deallocate(this%bifurcation_complete)
    !end if
    !if (associated(this%bifurcated_next_cell_index)) then
    !  deallocate(this%bifurcated_next_cell_index)
    !end if
    call this%destructor
end subroutine icon_single_index_destructor

subroutine generate_cumulative_flow(this,set_links)
  class(flow_accumulation_algorithm), intent(inout) :: this
  logical, intent(in) :: set_links
    call this%dependencies%for_all(set_dependencies_wrapper,this)
    call this%dependencies%for_all(add_cells_to_queue_wrapper,this)
    call this%process_queue()
    if (this%search_for_loops) call this%dependencies%for_all(check_for_loops_wrapper,this)
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

subroutine check_for_loops_wrapper(this,cell_coords)
  class(*), intent(inout) :: this
  class(coords), pointer, intent(inout) :: cell_coords
  class(*), pointer :: dependency_ptr
    select type(this)
    class is (flow_accumulation_algorithm)
      dependency_ptr => this%dependencies%get_value(cell_coords)
      select type(dependency_ptr)
      type is (integer)
        if (dependency_ptr /= 0) then
          call this%label_loop(cell_coords)
        end if
      end select
    end select
    deallocate(dependency_ptr)
    deallocate(cell_coords)
end subroutine check_for_loops_wrapper

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

subroutine label_loop(this,start_coords)
  class(flow_accumulation_algorithm), intent(in) :: this
  class(coords), pointer :: start_coords
  class(coords), pointer :: current_coords
  class(coords), pointer :: new_current_coords
    allocate(current_coords,source=start_coords)
    do
      call this%dependencies%set_value(current_coords,0)
      call this%cumulative_flow%set_value(current_coords,0)
      new_current_coords => this%get_next_cell_coords(current_coords)
      deallocate(current_coords)
      if (new_current_coords%are_equal_to(start_coords)) then
        deallocate(new_current_coords)
        exit
      end if
      current_coords => new_current_coords
    end do
end subroutine label_loop

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
      deallocate(next_cell_coords_lat_ptr)
      deallocate(next_cell_coords_lon_ptr)
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

  subroutine update_bifurcated_flows(this)
    class(flow_accumulation_algorithm), intent(inout) :: this
      call this%bifurcation_complete(1)%ptr%for_all(check_for_bifurcations_in_cell_wrapper,this)
  end subroutine

  subroutine check_for_bifurcations_in_cell_wrapper(this,coords_in)
    class(*), intent(inout) :: this
    class(coords), pointer, intent(inout) :: coords_in
      select type(this)
      class is (flow_accumulation_algorithm)
        call this%check_for_bifurcations_in_cell(coords_in)
      end select
  end subroutine check_for_bifurcations_in_cell_wrapper

  subroutine check_for_bifurcations_in_cell(this,coords_in)
    class(flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer, intent(inout) :: coords_in
    class(*), pointer :: cumulative_flow_value_ptr
    class(coords), pointer :: target_coords
    integer :: i
      if (this%is_bifurcated(coords_in)) then
        do i=1,this%max_neighbors - 1
          if(this%is_bifurcated(coords_in,i)) then
            target_coords => this%get_next_cell_bifurcated_coords(coords_in,i)
            cumulative_flow_value_ptr => this%cumulative_flow%get_value(coords_in)
            select type (cumulative_flow_value_ptr)
              type is (integer)
                call this%update_bifurcated_flow(target_coords, &
                                                 cumulative_flow_value_ptr)
            end select
            call this%bifurcation_complete(i)%ptr%set_value(coords_in,.true.)
          end if
        end do
      end if
  end subroutine check_for_bifurcations_in_cell


  recursive subroutine update_bifurcated_flow(this,initial_coords,additional_accumulated_flow)
  class(flow_accumulation_algorithm), intent(inout) :: this
  class(coords), pointer, intent(inout) :: initial_coords
  integer, intent(in) :: additional_accumulated_flow
  class(coords), pointer :: current_coords
  class(coords), pointer :: target_coords
  class(*), pointer :: cumulative_flow_value_ptr
  class(*), pointer :: bifurcated_complete_ptr
  integer :: i
    allocate(current_coords,source=initial_coords)
    do
      cumulative_flow_value_ptr => this%cumulative_flow%get_value(current_coords)
      select type (cumulative_flow_value_ptr)
        type is (integer)
          call this%cumulative_flow%set_value(current_coords, &
                                              cumulative_flow_value_ptr + &
                                              additional_accumulated_flow)
      end select
      if (this%is_bifurcated(current_coords)) then
        do i=1,this%max_neighbors - 1
          bifurcated_complete_ptr => this%bifurcation_complete(i)%ptr%get_value(current_coords)
          select type (bifurcated_complete_ptr)
            type is (logical)
              if(this%is_bifurcated(current_coords,i) .and. bifurcated_complete_ptr) then
                target_coords => this%get_next_cell_bifurcated_coords(current_coords,i)
                call this%update_bifurcated_flow(target_coords, &
                                                 additional_accumulated_flow)
            end if
          end select
        end do
      end if
      if (current_coords%are_equal_to(this%get_flow_terminates_value())) exit
      target_coords => this%get_next_cell_coords(current_coords)
      deallocate(current_coords)
      current_coords => target_coords
    end do
  end subroutine

  function latlon_is_bifurcated(this,coords_in,layer_in) result(bifurcated)
    class(latlon_flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer :: coords_in
    integer, optional, intent(in) :: layer_in
    class(*),pointer :: bifurcated_next_cell_index_lat_value
    logical :: bifurcated
    integer :: i
      if (present(layer_in)) then
        bifurcated_next_cell_index_lat_value => &
          this%bifurcated_next_cell_index_lat(layer_in)%ptr%get_value(coords_in)
        select type(bifurcated_next_cell_index_lat_value)
          type is (integer)
            bifurcated = (bifurcated_next_cell_index_lat_value &
                         /= this%no_bifurcation_value)
        end select
      else
        bifurcated = .false.
        do i = 1,this%max_neighbors - 1
          bifurcated_next_cell_index_lat_value => &
            this%bifurcated_next_cell_index_lat(i)%ptr%get_value(coords_in)
          select type(bifurcated_next_cell_index_lat_value)
            type is (integer)
              bifurcated = bifurcated .or. &
                           (bifurcated_next_cell_index_lat_value &
                           /= this%no_bifurcation_value)
          end select
          if (bifurcated) exit
        end do
      end if
  end function latlon_is_bifurcated

  function icon_single_index_is_bifurcated(this,coords_in,layer_in) result(bifurcated)
    class(icon_single_index_flow_accumulation_algorithm), intent(inout) :: this
    class(coords), pointer :: coords_in
    integer, optional, intent(in) :: layer_in
    class(*), pointer :: bifurcated_next_cell_index_value_ptr
    integer :: i
    logical :: bifurcated
      if (present(layer_in)) then
        bifurcated_next_cell_index_value_ptr => &
          this%bifurcated_next_cell_index(layer_in)%ptr%get_value(coords_in)
        select type (bifurcated_next_cell_index_value_ptr)
          type is (integer)
            bifurcated = (bifurcated_next_cell_index_value_ptr &
                          /= this%no_bifurcation_value)
        end select
      else
        bifurcated = .false.
        do i = 1,this%max_neighbors - 1
          bifurcated_next_cell_index_value_ptr => &
            this%bifurcated_next_cell_index(i)%ptr%get_value(coords_in)
          select type (bifurcated_next_cell_index_value_ptr)
            type is (integer)
            bifurcated = bifurcated .or. &
                         (bifurcated_next_cell_index_value_ptr &
                         /= this%no_bifurcation_value)
          end select
          if (bifurcated) exit
        end do
      end if
  end function icon_single_index_is_bifurcated

  function latlon_get_next_cell_bifurcated_coords(this,coords_in,layer_in) &
    result(next_cell_coords)
    class(latlon_flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: coords_in
    class(coords), pointer :: next_cell_coords
    class(*), pointer :: bifurcated_next_cell_coords_lat_ptr
    class(*), pointer :: bifurcated_next_cell_coords_lon_ptr
    integer, intent(in) :: layer_in
      bifurcated_next_cell_coords_lat_ptr => &
        this%bifurcated_next_cell_index_lat(layer_in)%ptr%get_value(coords_in)
      bifurcated_next_cell_coords_lon_ptr => &
        this%bifurcated_next_cell_index_lon(layer_in)%ptr%get_value(coords_in)
      select type (bifurcated_next_cell_coords_lat_ptr)
      type is (integer)
        select type (bifurcated_next_cell_coords_lon_ptr)
        type is (integer)
          allocate(next_cell_coords, &
                   source=latlon_coords(bifurcated_next_cell_coords_lat_ptr, &
                                        bifurcated_next_cell_coords_lon_ptr))
        end select
      end select
  end function latlon_get_next_cell_bifurcated_coords

  function icon_single_index_get_next_cell_bifurcated_coords(this,coords_in,layer_in) &
      result(next_cell_coords)
    class(icon_single_index_flow_accumulation_algorithm), intent(in) :: this
    class(coords), pointer :: coords_in
    class(coords), pointer :: next_cell_coords
    class(*), pointer :: bifurcated_next_cell_coords_index_ptr
    integer, intent(in) :: layer_in
      bifurcated_next_cell_coords_index_ptr => &
        this%bifurcated_next_cell_index(layer_in)%ptr%get_value(coords_in)
      select type (bifurcated_next_cell_coords_index_ptr)
        type is (integer)
          allocate(next_cell_coords, &
                   source=generic_1d_coords(bifurcated_next_cell_coords_index_ptr,.true.))
      end select
      deallocate(bifurcated_next_cell_coords_index_ptr)
  end function icon_single_index_get_next_cell_bifurcated_coords

end module flow_accumulation_algorithm_mod
