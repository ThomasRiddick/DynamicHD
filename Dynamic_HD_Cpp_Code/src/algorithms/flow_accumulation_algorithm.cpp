// subroutine init_flow_accumulation_algorithm(this)
//     call this%dependencies%set_all(0)
//     call this%cumulative_flow%set_all(0)
// }

// subroutine latlon_init_flow_accumulation_algorithm( &
//                                                    next_cell_index_lat, &
//                                                    next_cell_index_lon, &
//                                                    cumulative_flow)
//   class(*), dimension(:,:), pointer, intent(in) :: next_cell_index_lat
//   class(*), dimension(:,:), pointer, intent(in) :: next_cell_index_lon
//   class(*), dimension(:,:), pointer, intent(inout) :: cumulative_flow
//   class(*), dimension(:,:), pointer :: dependencies_data
//   type(latlon_section_coords), allocatable :: field_section_coords
//     allocate(field_section_coords)
//     field_section_coords = latlon_section_coords(1,1,size(next_cell_index_lat,1), &
//                                                      size(next_cell_index_lat,2))
//     this%next_cell_index_lat => latlon_subfield(next_cell_index_lat,field_section_coords,true)
//     this%next_cell_index_lon => latlon_subfield(next_cell_index_lon,field_section_coords,true)
//     this%cumulative_flow => latlon_subfield(cumulative_flow,field_section_coords,true)
//     allocate(integer::dependencies_data(size(next_cell_index_lat,1),&
//                                         size(next_cell_index_lon,2)))
//     this%dependencies => latlon_subfield(dependencies_data, &
//                                          field_section_coords,true)
//     allocate(this%external_data_value,source=latlon_coords(-1,-1))
//     allocate(this%flow_terminates_value,source=latlon_coords(-2,-2))
//     allocate(this%no_data_value,source=latlon_coords(-3,-3))
//     allocate(this%no_flow_value,source=latlon_coords(-4,-4))
//     call this%init_flow_accumulation_algorithm()
// }

// subroutine icon_single_index_init_flow_accumulation_algorithm(field_section_coords, &
//                                                               next_cell_index, &
//                                                               cumulative_flow, &
//                                                               bifurcated_next_cell_index)
//   class(*), dimension(:), pointer, intent(in) :: next_cell_index
//   class(*), dimension(:), pointer, intent(out) :: cumulative_flow
//   class(*), dimension(:,:), pointer, intent(in), optional :: bifurcated_next_cell_index
//   class(*), dimension(:), pointer :: bifurcated_next_cell_index_slice
//   class(*), dimension(:), pointer :: bifurcation_complete_slice
//    type(generic_1d_section_coords), intent(in) :: field_section_coords
//   class(*), dimension(:), pointer :: dependencies_data
//   integer :: i
//     this%max_neighbors = 12
//     this%no_bifurcation_value = -9
//     this%next_cell_index => icon_single_index_subfield(next_cell_index,field_section_coords)
//     this%cumulative_flow => icon_single_index_subfield(cumulative_flow,field_section_coords)
//     allocate(integer::dependencies_data(size(next_cell_index)))
//     this%dependencies => icon_single_index_subfield(dependencies_data, &
//                                                     field_section_coords)
//     allocate(this%external_data_value,source=generic_1d_coords(-2,true))
//     allocate(this%flow_terminates_value,source=generic_1d_coords(-3,true))
//     allocate(this%no_data_value,source=generic_1d_coords(-4,true))
//     allocate(this%no_flow_value,source=generic_1d_coords(-5,true))
//     if (present(bifurcated_next_cell_index)) {
//       allocate(subfield_ptr::this%bifurcated_next_cell_index(this%max_neighbors-1))
//       allocate(subfield_ptr::this%bifurcation_complete(this%max_neighbors-1))
//       do i = 1,this%max_neighbors - 1 {
//         bifurcated_next_cell_index_slice => bifurcated_next_cell_index(:,i)
//         this%bifurcated_next_cell_index(i)%ptr => &
//           icon_single_index_subfield(bifurcated_next_cell_index_slice,field_section_coords)
//         allocate(logical::bifurcation_complete_slice(size(next_cell_index)))
//         select type(bifurcation_complete_slice)
//           type is (logical)
//             bifurcation_complete_slice(:) = false
//         end select
//         this%bifurcation_complete(i)%ptr => &
//           icon_single_index_subfield(bifurcation_complete_slice,field_section_coords)
//       }
//     }
//     call this%init_flow_accumulation_algorithm()
// }

void flow_accumulation_algorithm::generate_cumulative_flow(bool set_links){
    dependencies->for_all(set_dependencies);
    dependencies->for_all(add_cells_to_queue);
    process_queue();
    if (search_for_loops) dependencies->for_all(check_for_loops_wrapper)
    if (set_links) dependencies->for_all_edge_cells(follow_paths_wrapper)
}

void flow_accumulation_algorithm::set_dependencies(coords* coords_in){
  class(coords), pointer, intent(inout) :: coords_in
  class(coords),pointer :: target_coords
  class(coords),pointer :: target_of_target_coords
  class(*), pointer  :: dependency_ptr
    if ( .not. (coords_in%are_equal_to(this%get_no_data_value()) .or. &
                coords_in%are_equal_to(this%get_no_flow_value()) .or. &
                coords_in%are_equal_to(this%get_flow_terminates_value()) )) {
      target_coords => this%get_next_cell_coords(coords_in)
      if ( target_coords%are_equal_to(this%get_no_data_value()) ) {
        call this%cumulative_flow%set_value(coords_in,this%get_no_data_value())
      } else if ( target_coords%are_equal_to(this%get_no_flow_value())) {
        call this%cumulative_flow%set_value(coords_in,this%get_no_flow_value())
      } else if ( .not. target_coords%are_equal_to(this%get_flow_terminates_value())) {
        target_of_target_coords => this%get_next_cell_coords(target_coords)
        if ( .not. (target_of_target_coords%are_equal_to(this%get_no_flow_value()) .or. &
                    target_of_target_coords%are_equal_to(this%get_no_data_value()) )) {
          dependency_ptr => this%dependencies%get_value(target_coords)
          select type (dependency_ptr)
            type is (integer)
              call this%dependencies%set_value(target_coords, &
                                               dependency_ptr+1)
          end select
          deallocate(dependency_ptr)
        }
        deallocate(target_of_target_coords)
      }
      deallocate(target_coords)
    }
    deallocate(coords_in)
}

void flow_accumulation_algorithm::add_cells_to_queue(coords* coords_in){
  class(coords), pointer, intent(inout) :: coords_in
  class(coords), pointer :: target_coords
  class(*), pointer  :: dependency_ptr
  target_coords => this%get_next_cell_coords(coords_in)
  dependency_ptr => this%dependencies%get_value(coords_in)
  select type (dependency_ptr)
    type is (integer)
    if (  dependency_ptr == 0 .and. &
         ( .not. target_coords%are_equal_to(this%get_no_data_value()) ) ) {
      call this%q%add_value_to_back(coords_in)
      if ( .not. target_coords%are_equal_to(this%get_flow_terminates_value())) {
        call this%cumulative_flow%set_value(coords_in,1)
      }
    }
  end select
  deallocate(coords_in)
  deallocate(dependency_ptr)
  deallocate(target_coords)
}

void flow_accumulation_algorithm::process_queue(){
  class(coords), pointer :: current_coords
  class(*),      pointer :: current_coords_ptr
  class(coords), pointer :: target_coords
  class(coords), pointer :: target_of_target_coords
  class(*), pointer :: cumulative_flow_current_coords_ptr
  class(*), pointer :: cumulative_flow_target_coords_ptr
  class(*), pointer :: dependency_ptr
  integer :: dependency
  do while ( .not. this%q%iterate_forward() ) {
    current_coords_ptr => this%q%get_value_at_iterator_position()
    select type(current_coords_ptr)
      class is (coords)
        current_coords => current_coords_ptr
    end select
    target_coords => this%get_next_cell_coords(current_coords)
    if ( target_coords%are_equal_to(this%get_no_data_value()) .or. &
         target_coords%are_equal_to(this%get_no_flow_value()) .or. &
         target_coords%are_equal_to(this%get_flow_terminates_value())) {
      call this%q%remove_element_at_iterator_position()
      deallocate(target_coords)
      cycle
    }
    target_of_target_coords => this%get_next_cell_coords(target_coords)
    if ( target_of_target_coords%are_equal_to(this%get_no_data_value())) {
      call this%q%remove_element_at_iterator_position()
      deallocate(target_of_target_coords)
      cycle
    }
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
                target_coords%are_equal_to(this%get_flow_terminates_value())) {
              call this%cumulative_flow%set_value(target_coords, &
                                                  cumulative_flow_target_coords_ptr + &
                                                  cumulative_flow_current_coords_ptr + 1)
              call this%q%add_value_to_back(target_coords)
            } else {
              call this%cumulative_flow%set_value(target_coords, &
                                                  cumulative_flow_target_coords_ptr + &
                                                  cumulative_flow_current_coords_ptr)
            }
        end select
      end select
      deallocate(target_coords)
      deallocate(cumulative_flow_target_coords_ptr)
      deallocate(cumulative_flow_current_coords_ptr)
  }
}

void flow_accumulation_algorithm::check_for_loops_wrapper(cell_coords){
  class(coords), pointer, intent(inout) :: cell_coords
  class(*), pointer :: dependency_ptr
    select type(this)
    class is (flow_accumulation_algorithm)
      dependency_ptr => this%dependencies%get_value(cell_coords)
      select type(dependency_ptr)
      type is (integer)
        if (dependency_ptr /= 0) {
          call this%label_loop(cell_coords)
        }
      end select
    end select
    deallocate(dependency_ptr)
    deallocate(cell_coords)
}

void flow_accumulation_algorithm::follow_paths_wrapper(initial_coords){
  class(coords), pointer, intent(inout) :: initial_coords
    select type(this)
    class is (flow_accumulation_algorithm)
      call this%follow_paths(initial_coords)
    end select
}

void flow_accumulation_algorithm::follow_paths(initial_coords){
  class(coords), pointer, intent(inout) :: initial_coords
  class(coords), pointer :: current_coords
  class(coords), pointer :: target_coords
  integer :: coords_index
  allocate(current_coords,source=initial_coords)
  coords_index = this%generate_coords_index(initial_coords)
  do {
    if ( target_coords%are_equal_to(this%get_no_data_value()) .or. &
         target_coords%are_equal_to(this%get_no_flow_value()) ) {
      call this%assign_coords_to_link_array(coords_index,this%get_flow_terminates_value())
      exit
    }
    target_coords => this%get_next_cell_coords(current_coords)
    if ( this%dependencies%coords_outside_subfield(target_coords) ) {
      if ( current_coords%are_equal_to(initial_coords) ) {
         call this%assign_coords_to_link_array(coords_index,this%get_external_flow_value())
      } else {
         call this%assign_coords_to_link_array(coords_index,current_coords)
      }
      exit
    }
    current_coords => target_coords
  }
  deallocate(initial_coords)
}

void flow_accumulation_algorithm::label_loop(start_coords){
  class(coords), pointer :: start_coords
  class(coords), pointer :: current_coords
  class(coords), pointer :: new_current_coords
    allocate(current_coords,source=start_coords)
    do {
      call this%dependencies%set_value(current_coords,0)
      call this%cumulative_flow%set_value(current_coords,0)
      new_current_coords => this%get_next_cell_coords(current_coords)
      deallocate(current_coords)
      if (new_current_coords%are_equal_to(start_coords)) {
        deallocate(new_current_coords)
        exit
      }
      current_coords => new_current_coords
    }
}

coords* flow_accumulation_algorithm::get_external_flow_value() {
    external_data_value => this%external_data_value
}

coords* flow_accumulation_algorithm::get_flow_terminates_value() {
    flow_terminates_value => this%flow_terminates_value
}

coords* flow_accumulation_algorithm::get_no_data_value() {
    no_data_value => this%no_data_value
}

coords* flow_accumulation_algorithm::get_no_flow_value() {
    no_flow_value => this%no_flow_value
}

coords* latlon_flow_accumulation_algorithm::
  get_next_cell_coords(coords* coords_in) {
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
}

int latlon_flow_accumulation_algorithm::
  generate_coords_index(coords* coords_in) {
  integer :: index
  integer :: relative_lat, relative_lon
  select type (coords_in)
  class is (latlon_coords)
    relative_lat = 1 + coords_in%lat - this%tile_min_lat
    relative_lon = 1 + coords_in%lon - this%tile_min_lon
  end select
  if ( relative_lat == 1 ) {
    index = relative_lon
  } else if ( relative_lat < this%tile_width_lat .and. &
            relative_lat > 1 ) {
    if ( relative_lon == 1 ) {
      index = 2*this%tile_width_lon + 2*this%tile_width_lat - relative_lat - 2
    } else if ( relative_lon == this%tile_width_lon ) {
      index = this%tile_width_lon + relative_lat - 1
    } else {
      stop 'trying to generate the index of an non edge cell'
    }
  } else if ( relative_lat == this%tile_width_lat ) {
    index = this%tile_width_lat + 2*this%tile_width_lon - relative_lon - 1
  } else {
      stop 'trying to generate the index of an non edge cell'
  }
}


void latlon_flow_accumulation_algorithm::
  assign_coords_to_link_array(int coords_index,coords* coords_in){
  select type(latlon_links => this%links)
    type is (latlon_coords)
    select type(coords_in)
      type is (latlon_coords)
        latlon_links(coords_index) = coords_in
    end select
  end select
}

coords* icon_single_index_flow_accumulation_algorithm::
  get_next_cell_coords(coords* coords_in) {
  class(coords), pointer :: next_cell_coords
  class(*), pointer :: next_cell_coords_index_ptr
    next_cell_coords_index_ptr => this%next_cell_index%get_value(coords_in)
        allocate(next_cell_coords, &
                 source=generic_1d_coords(next_cell_coords_index_ptr,true))
}

int icon_single_index_flow_accumulation_algorithm::
  generate_coords_index(coords* coords_in) {
    index = 1
    stop "Function generate_coords_index not yet implemented for incon grid"
}

void icon_single_index_flow_accumulation_algorithm::
  assign_coords_to_link_array(coords* coords_index,coords* coords_in){
  select type(generic_1d_links => this%links)
    type is (generic_1d_coords)
    select type(coords_in)
      type is (generic_1d_coords)
        generic_1d_links(coords_index) = coords_in
    end select
  end select
}

void flow_accumulation_algorithm::update_bifurcated_flows(){
    call this%bifurcation_complete(1)%ptr%for_all(check_for_bifurcations_in_cell_wrapper,this)
}

void flow_accumulation_algorithm::
  check_for_bifurcations_in_cell_wrapper(coords* coords_in){
    select type(this)
    class is (flow_accumulation_algorithm)
      call this%check_for_bifurcations_in_cell(coords_in)
    end select
}

void flow_accumulation_algorithm::
  check_for_bifurcations_in_cell(coords* coords_in){
  class(*), pointer :: cumulative_flow_value_ptr
  class(coords), pointer :: target_coords
  integer :: i
    if (this%is_bifurcated(coords_in)) {
      do i=1,this%max_neighbors - 1 {
        if(this%is_bifurcated(coords_in,i)) {
          target_coords => this%get_next_cell_bifurcated_coords(coords_in,i)
          cumulative_flow_value_ptr => this%cumulative_flow%get_value(coords_in)
          select type (cumulative_flow_value_ptr)
            type is (integer)
              call this%update_bifurcated_flow(target_coords, &
                                               cumulative_flow_value_ptr)
          end select
          call this%bifurcation_complete(i)%ptr%set_value(coords_in,true)
        }
      }
    }
}

void flow_accumulation_algorithm::
  update_bifurcated_flow(coords* initial_coords,
                         int additional_accumulated_flow){
class(coords), pointer :: current_coords
class(coords), pointer :: target_coords
class(*), pointer :: cumulative_flow_value_ptr
class(*), pointer :: bifurcated_complete_ptr
integer :: i
  allocate(current_coords,source=initial_coords)
  do {
    cumulative_flow_value_ptr => this%cumulative_flow%get_value(current_coords)
    select type (cumulative_flow_value_ptr)
      type is (integer)
        call this%cumulative_flow%set_value(current_coords, &
                                            cumulative_flow_value_ptr + &
                                            additional_accumulated_flow)
    end select
    if (this%is_bifurcated(current_coords)) {
      do i=1,this%max_neighbors - 1 {
        bifurcated_complete_ptr => this%bifurcation_complete(i)%ptr%get_value(current_coords)
        select type (bifurcated_complete_ptr)
          type is (logical)
            if(this%is_bifurcated(current_coords,i) .and. bifurcated_complete_ptr) {
              target_coords => this%get_next_cell_bifurcated_coords(current_coords,i)
              call this%update_bifurcated_flow(target_coords, &
                                               additional_accumulated_flow)
          }
        end select
      }
    }
    if (current_coords%are_equal_to(this%get_flow_terminates_value())) exit
    target_coords => this%get_next_cell_coords(current_coords)
    deallocate(current_coords)
    current_coords => target_coords
  }
}

bool latlon_flow_accumulation_algorithm::is_bifurcated(coords* coords_in,
                                                       int layer_in = - 1){
  class(*),pointer :: bifurcated_next_cell_index_lat_value
  logical :: bifurcated
  integer :: i
    if (present(layer_in)) {
      bifurcated_next_cell_index_lat_value => &
        this%bifurcated_next_cell_index_lat(layer_in)%ptr%get_value(coords_in)
      select type(bifurcated_next_cell_index_lat_value)
        type is (integer)
          bifurcated = (bifurcated_next_cell_index_lat_value &
                       /= this%no_bifurcation_value)
      end select
    } else {
      bifurcated = false
      do i = 1,this%max_neighbors - 1 {
        bifurcated_next_cell_index_lat_value => &
          this%bifurcated_next_cell_index_lat(i)%ptr%get_value(coords_in)
        select type(bifurcated_next_cell_index_lat_value)
          type is (integer)
            bifurcated = bifurcated .or. &
                         (bifurcated_next_cell_index_lat_value &
                         /= this%no_bifurcation_value)
        end select
        if (bifurcated) exit
      }
    }
}

bool icon_single_index_flow_accumulation_algorithm::
  is_bifurcated(coords* coords_in,int layer_in = -1){
  class(*), pointer :: bifurcated_next_cell_index_value_ptr
  integer :: i
    if (present(layer_in)) {
      bifurcated_next_cell_index_value_ptr => &
        this%bifurcated_next_cell_index(layer_in)%ptr%get_value(coords_in)
      select type (bifurcated_next_cell_index_value_ptr)
        type is (integer)
          bifurcated = (bifurcated_next_cell_index_value_ptr &
                        /= this%no_bifurcation_value)
      end select
    } else {
      bifurcated = false
      do i = 1,this%max_neighbors - 1 {
        bifurcated_next_cell_index_value_ptr => &
          this%bifurcated_next_cell_index(i)%ptr%get_value(coords_in)
        select type (bifurcated_next_cell_index_value_ptr)
          type is (integer)
          bifurcated = bifurcated .or. &
                       (bifurcated_next_cell_index_value_ptr &
                       /= this%no_bifurcation_value)
        end select
        if (bifurcated) exit
      }
    }
}

coords* latlon_flow_accumulation_algorithm::
  get_next_cell_bifurcated_coords(coords* coords_in,
                                  int layer_in) {
  class(*), pointer :: bifurcated_next_cell_coords_lat_ptr
  class(*), pointer :: bifurcated_next_cell_coords_lon_ptr
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
}

coords* icon_single_index_flow_accumulation_algorithm::
  get_next_cell_bifurcated_coords(coords* coords_in,
                                  int layer_in) {
  class(*), pointer :: bifurcated_next_cell_coords_index_ptr
  integer, intent(in) :: layer_in
    bifurcated_next_cell_coords_index_ptr => &
      this%bifurcated_next_cell_index(layer_in)%ptr%get_value(coords_in)
    select type (bifurcated_next_cell_coords_index_ptr)
      type is (integer)
        allocate(next_cell_coords, &
                 source=generic_1d_coords(bifurcated_next_cell_coords_index_ptr,true))
    end select
    deallocate(bifurcated_next_cell_coords_index_ptr)
}
