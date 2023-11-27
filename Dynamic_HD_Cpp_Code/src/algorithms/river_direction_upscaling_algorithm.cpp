    void find_next_pixel_upstream(coords* coords_inout,
                                  OUT bool coords_not_in_area){
        vector<coords*> upstream_neighbors_coords_list;
        class(*), pointer :: upstream_neighbor_total_cumulative_flow
        int max_cmltv_flow_working_value
        int i
        max_cmltv_flow_working_value = 0
        upstream_neighbors_coords_list => this%find_upstream_neighbors(coords_inout)
        if (size(upstream_neighbors_coords_list) > 0) {
            do i = 1, size(upstream_neighbors_coords_list) {
                upstream_neighbor_total_cumulative_flow =>
                    this%total_cumulative_flow%get_value(upstream_neighbors_coords_list(i))
                select type (upstream_neighbor_total_cumulative_flow)
                type is (integer)
                    if ( upstream_neighbor_total_cumulative_flow > max_cmltv_flow_working_value) {
                        if (allocated(coords_inout)) deallocate(coords_inout)
                        allocate(coords_inout,source=upstream_neighbors_coords_list(i))
                        max_cmltv_flow_working_value = upstream_neighbor_total_cumulative_flow
                    }
                end select
                deallocate(upstream_neighbor_total_cumulative_flow)
            }
            coords_not_in_area = .not. this%check_if_coords_are_in_area(coords_inout)
        } else {
            coords_not_in_area = true
        }
        deallocate(upstream_neighbors_coords_list)
    }

    void field_destructor(){
            if (associated(this%cells_to_reprocess)) call this%cells_to_reprocess%deallocate_data()
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (associated(this%cells_to_reprocess)) deallocate(this%cells_to_reprocess)
    }

    bool check_cell_for_localized_loops(coords* coords_inout){
        class(coords),allocatable :: initial_cell_coords
        bool coords_not_in_area_dummy
        bool is_sea_point_or_horizontal_edge_or_sink
            allocate(initial_cell_coords,source=coords_inout)
            call this%find_next_pixel_downstream(coords_inout,coords_not_in_area_dummy)
            if (coords_inout%are_equal_to(initial_cell_coords)) {
                is_sea_point_or_horizontal_edge_or_sink = true
            } else {
                is_sea_point_or_horizontal_edge_or_sink = false
            }
            call this%find_next_pixel_downstream(coords_inout,coords_not_in_area_dummy)
            if (coords_inout%are_equal_to(initial_cell_coords) .and.
                .not. is_sea_point_or_horizontal_edge_or_sink) {
                localized_loop_found = true
            } else {
                localized_loop_found = false
            }
            deallocate(initial_cell_coords)
    }

    coords* find_outlet_pixel(OUT bool no_remaining_outlets,
                               bool use_LCDA_criterion,
                               OUT bool outlet_is_LCDA){
        logical, intent(inout) :: no_remaining_outlets
        logical, intent(in)    :: use_LCDA_criterion
        logical, intent(inout), optional :: outlet_is_LCDA
        class(coords), pointer :: LUDA_pixel_coords
        class(coords), pointer :: LCDA_pixel_coords => null()
            nullify(LUDA_pixel_coords)
            if(use_LCDA_criterion) {
                call this%generate_cell_cumulative_flow()
                LCDA_pixel_coords=>this%find_pixel_with_LCDA()
            } else {
                nullify(LCDA_pixel_coords)
            }
            do {
                if (associated(LUDA_pixel_coords)) deallocate(LUDA_pixel_coords)
                LUDA_pixel_coords=>this%find_pixel_with_LUDA(no_remaining_outlets)
                if (no_remaining_outlets) {
                    if (present(outlet_is_LCDA)) outlet_is_LCDA = false
                    exit
                } else {
                    if (present(outlet_is_LCDA)) {
                        if (this%check_MUFP(LUDA_pixel_coords,LCDA_pixel_coords,outlet_is_LCDA)) exit
                    } else {
                        if (this%check_MUFP(LUDA_pixel_coords,LCDA_pixel_coords)) exit
                    }
                }
            }
            if (use_LCDA_criterion) deallocate(LCDA_pixel_coords)
        return LUDA_pixel_coords;
    }

    void check_for_sinks_and_rmouth_outflows(coords* outlet_pixel_coords,
                                             bool no_remaining_outlets,
                                             OUT bool mark_sink,
                                             OUT bool mark_outflow){
        logical, intent(in) :: no_remaining_outlets
        logical, intent(out) :: mark_sink
        logical, intent(out) :: mark_outflow
        class(*), pointer :: outlet_pixel_cumulative_flow_value
        class(coords), pointer :: main_sink_coords
        class(coords), pointer :: main_river_mouth_coords
        int outlet_pixel_cumulative_flow
        int cumulative_sink_outflow
        int cumulative_rmouth_outflow
            mark_sink = false
            mark_outflow = false
            if (no_remaining_outlets) {
                outlet_pixel_cumulative_flow = -1
            } else {
                outlet_pixel_cumulative_flow_value =>
                    this%total_cumulative_flow%get_value(outlet_pixel_coords)
                select type(outlet_pixel_cumulative_flow_value)
                type is (integer)
                    outlet_pixel_cumulative_flow = outlet_pixel_cumulative_flow_value
                end select
                deallocate(outlet_pixel_cumulative_flow_value)
            }
            if (run_check_for_sinks) {
                cumulative_sink_outflow =
                    this%get_sink_combined_cumulative_flow(main_sink_coords)
            } else {
                cumulative_sink_outflow = 0
            }
            if (this%contains_river_mouths) {
                cumulative_rmouth_outflow =
                    this%get_rmouth_outflow_combined_cumulative_flow(main_river_mouth_coords)
            } else {
                cumulative_rmouth_outflow = -1
            }
            if ( cumulative_sink_outflow > cumulative_rmouth_outflow .and.
                 cumulative_sink_outflow > outlet_pixel_cumulative_flow ) {
                 mark_sink = true
                 deallocate(outlet_pixel_coords)
                 if (this%contains_river_mouths) deallocate(main_river_mouth_coords)
                 outlet_pixel_coords =>  main_sink_coords
            } else if (  cumulative_rmouth_outflow > outlet_pixel_cumulative_flow) {
                 mark_outflow = true
                 deallocate(outlet_pixel_coords)
                 if (run_check_for_sinks) deallocate(main_sink_coords)
                 outlet_pixel_coords =>  main_river_mouth_coords
            } else {
                if (this%contains_river_mouths) deallocate(main_river_mouth_coords)
                if (run_check_for_sinks) deallocate(main_sink_coords)
            }
    }


    direction_indicator* find_cell_flow_direction(coords* outlet_pixel_coords,
                                                  bool no_remaining_outlets) {
        class(*), allocatable :: downstream_cell
        bool mark_sink;
        bool mark_outflow;
            call this%check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink,
                                                          mark_outflow)
            if (mark_sink) {
                flow_direction => this%get_flow_direction_for_sink(is_for_cell=true)
                return
            } else if (mark_outflow) {
                flow_direction => this%get_flow_direction_for_outflow(is_for_cell=true)
                return
            }
            allocate(downstream_cell,source=this%cell_neighborhood%find_downstream_cell(outlet_pixel_coords))
            select type (downstream_cell)
            type is (integer)
                flow_direction => this%cell_neighborhood%calculate_direction_indicator(downstream_cell)
            end select
    }

    direction_indicator* process_cell(){
        class(coords), pointer :: outlet_pixel_coords => null()
        bool no_remaining_outlets;
        bool use_LCDA_criterion = true;
            if (.not. this%ocean_cell) {
                outlet_pixel_coords => this%find_outlet_pixel(no_remaining_outlets,use_LCDA_criterion)
                flow_direction => this%find_cell_flow_direction(outlet_pixel_coords,no_remaining_outlets)
                deallocate(outlet_pixel_coords)
            } else {
                flow_direction => this%get_flow_direction_for_ocean_point(is_for_cell=true)
            }
    }

    void yamazaki_mark_cell_outlet_pixel(use_LCDA_criterion) {
        bool use_LCDA_criterion;
        class(coords), pointer :: outlet_pixel_coords => null()
        bool no_remaining_outlets
        bool mark_sink
        bool mark_outflow
        bool outlet_is_LCDA
        int outlet_type
            if (.not. this%ocean_cell) {
                outlet_is_LCDA = false
                outlet_pixel_coords => this%find_outlet_pixel(no_remaining_outlets,use_LCDA_criterion,
                                                              outlet_is_LCDA)
                call this%check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink,
                                                              mark_outflow)
                if (mark_sink) {
                    outlet_type = this%yamazaki_sink_outlet
                } else if (mark_outflow) {
                    outlet_type = this%yamazaki_rmouth_outlet
                } else {
                    if ( .not. (no_remaining_outlets .or. outlet_is_LCDA) ) {
                        outlet_type = this%yamazaki_normal_outlet
                    } else {
                        outlet_type = this%yamazaki_source_cell_outlet
                    }
                }
                call this%yamazaki_outlet_pixels%set_value(outlet_pixel_coords,outlet_type)
                deallocate(outlet_pixel_coords)
            }
    }

    void print_cell_and_neighborhood(){
        write(*,*) 'For cell'

        call this%print_area(this)
        write (*,*) 'contains_river_mouths= ', this%contains_river_mouths
        write (*,*) 'ocean_cell= ', this%ocean_cell
        write(*,*) 'For neighborhood'
        call this%cell_neighborhood%print_area(this%cell_neighborhood)
    }

    void cell_destructor() {
            if (associated(this%rejected_pixels)) {
                call this%rejected_pixels%destructor()
                deallocate(this%rejected_pixels)
            }
            if (associated(this%cell_cumulative_flow)) {
                call this%cell_cumulative_flow%destructor()
                deallocate(this%cell_cumulative_flow)
            }
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (allocated(this%cell_neighborhood)) call this%cell_neighborhood%destructor()
            if (associated(this%yamazaki_outlet_pixels)) deallocate(this%yamazaki_outlet_pixels)
    }

    coords* yamazaki_retrieve_initial_outlet_pixel(OUT int outlet_pixel_type) {
        class(coords), pointer :: working_pixel
        class(*), pointer :: working_pixel_ptr
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(doubly_linked_list), pointer :: pixels_to_process
        class(*), pointer :: working_pixel_value
        bool pixel_found
        int i
            pixel_found = false
            edge_pixel_coords_list => this%find_edge_pixels()
            do i = 1,size(edge_pixel_coords_list) {
                working_pixel => edge_pixel_coords_list(i)
                if ( .not. pixel_found) {
                    working_pixel_value => this%yamazaki_outlet_pixels%get_value(working_pixel)
                    select type(working_pixel_value)
                    type is (integer)
                        if (working_pixel_value == this%yamazaki_source_cell_outlet .or.
                            working_pixel_value == this%yamazaki_normal_outlet .or.
                            working_pixel_value == this%yamazaki_sink_outlet .or.
                            working_pixel_value == this%yamazaki_rmouth_outlet) {
                            outlet_pixel_type = working_pixel_value
                            allocate(initial_outlet_pixel,source=working_pixel)
                            pixel_found = true
                        }
                    end select
                    deallocate(working_pixel_value)
                }
            }
            deallocate(edge_pixel_coords_list)
            if ( .not. pixel_found) {
                pixels_to_process => this%generate_list_of_pixels_in_cell()
                do {
                    //Combine iteration and check to see if we have reached the end
                    //of the list
                    if (pixels_to_process%iterate_forward()) {
                        if (pixel_found) {
                            exit
                        } else {
                            stop 'no outlet pixel could be retrieved for cell'
                        }
                    }
                    working_pixel_ptr => pixels_to_process%get_value_at_iterator_position()
                    select type(working_pixel_ptr)
                        class is (coords)
                            working_pixel => working_pixel_ptr
                    end select
                    if ( .not. pixel_found) {
                        working_pixel_value => this%yamazaki_outlet_pixels%get_value(working_pixel)
                        select type(working_pixel_value)
                        type is (integer)
                            if (working_pixel_value == this%yamazaki_sink_outlet .or.
                                working_pixel_value == this%yamazaki_rmouth_outlet) {
                                outlet_pixel_type = working_pixel_value
                                allocate(initial_outlet_pixel,source=working_pixel)
                                pixel_found = true
                            }
                        end select
                        deallocate(working_pixel_value)
                    }
                    call pixels_to_process%remove_element_at_iterator_position()
                }
                deallocate(pixels_to_process)
            }
            return initial_outlet_pixel;
    }

    coords* find_pixel_with_LUDA(no_remaining_outlets) {
        coords* LUDA_pixel_coords;
        logical, intent(out) :: no_remaining_outlets
        class(coords), allocatable :: working_pixel_coords
        int LUDA_working_value
        int i
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(*), pointer :: rejected_pixel
        class(*), pointer :: upstream_drainage
        bool coords_not_in_cell
        nullify(LUDA_pixel_coords)
        edge_pixel_coords_list => this%find_edge_pixels()
            LUDA_working_value = 0
            do i = 1,size(edge_pixel_coords_list) {
                rejected_pixel => this%rejected_pixels%get_value(edge_pixel_coords_list(i))
                select type(rejected_pixel)
                type is (logical)
                    if ( .NOT. rejected_pixel) {
                        upstream_drainage => this%total_cumulative_flow%get_value(edge_pixel_coords_list(i))
                        select type (upstream_drainage)
                        type is (integer)
                        if (upstream_drainage > LUDA_working_value) {
                                allocate(working_pixel_coords,source=edge_pixel_coords_list(i))
                                call this%find_next_pixel_downstream(working_pixel_coords,coords_not_in_cell)
                                deallocate(working_pixel_coords)
                                if(coords_not_in_cell) {
                                    LUDA_working_value = upstream_drainage
                                    if (associated(LUDA_pixel_coords)) deallocate(LUDA_pixel_coords)
                                    allocate(LUDA_pixel_coords,source=edge_pixel_coords_list(i))
                                }
                            }
                        end select
                        deallocate(upstream_drainage)
                    }
                end select
                deallocate(rejected_pixel)
            }
            if (.not. associated(LUDA_pixel_coords)) {
                allocate(LUDA_pixel_coords,source=edge_pixel_coords_list(1))
                no_remaining_outlets=true
            } else {
                no_remaining_outlets=false
            }
            deallocate(edge_pixel_coords_list)
            return LUDA_pixel_coords;
    }

    coords* find_pixel_with_LCDA() {
        coords* LCDA_pixel_coords;
        class(coords), allocatable :: working_pixel_coords
        int LCDA_pixel_working_value;
        class(coords), pointer :: edge_pixel_coords_list(:)
        class(*), pointer :: cell_drainage_area
        bool coords_not_in_cell
        int i
            nullify(LCDA_pixel_coords)
            edge_pixel_coords_list => this%find_edge_pixels()
            LCDA_pixel_working_value = 0
            do i = 1,size(edge_pixel_coords_list) {
                cell_drainage_area => this%cell_cumulative_flow%get_value(edge_pixel_coords_list(i))
                select type(cell_drainage_area)
                type is (integer)
                    if (cell_drainage_area > LCDA_pixel_working_value) {
                        allocate(working_pixel_coords,source=edge_pixel_coords_list(i))
                        call this%find_next_pixel_downstream(working_pixel_coords,coords_not_in_cell)
                        deallocate(working_pixel_coords)
                        if(coords_not_in_cell) {
                            LCDA_pixel_working_value = cell_drainage_area
                            if (associated(LCDA_pixel_coords)) deallocate(LCDA_pixel_coords)
                            allocate(LCDA_pixel_coords,source=edge_pixel_coords_list(i))
                        }
                    }
                end select
                deallocate(cell_drainage_area)
            }
            if (.not. associated(LCDA_pixel_coords)) {
                allocate(LCDA_pixel_coords,source=edge_pixel_coords_list(1))
            }
            deallocate(edge_pixel_coords_list)
            return LCDA_pixel_coords;
    }

    double measure_upstream_path(coords* outlet_pixel_coords) {
        class(coords), allocatable :: working_coords
        bool coords_not_in_cell
        real(kind=double_precision) :: length_through_pixel
        real(kind=double_precision) :: upstream_path_length
            upstream_path_length = 0.0_double_precision
            allocate(working_coords, source=outlet_pixel_coords)
            do {
                length_through_pixel = this%calculate_length_through_pixel(working_coords)
                do {
                    upstream_path_length = upstream_path_length + length_through_pixel
                    call this%find_next_pixel_upstream(working_coords,coords_not_in_cell)
                    length_through_pixel = this%calculate_length_through_pixel(working_coords)
                    if (coords_not_in_cell) exit
                }
                if (.NOT. this%cell_neighborhood%path_reenters_region(working_coords)) exit
            }
            deallocate(working_coords)
            return upstream_path_length;
    }

    bool check_MUFP(coords* LUDA_pixel_coords,
                    coords* LCDA_pixel_coords,
                    OUT bool LUDA_is_LCDA) {
        bool accept_pixel;
            if (present(LUDA_is_LCDA)) LUDA_is_LCDA = false
            if (this%measure_upstream_path(LUDA_pixel_coords) > MUFP) {
                accept_pixel = true
                return
            } else if (associated(LCDA_pixel_coords)) {
                if(LUDA_pixel_coords%are_equal_to(LCDA_pixel_coords)) {
                    accept_pixel = true
                    if (present(LUDA_is_LCDA)) LUDA_is_LCDA = true
                    return
                }
            }
            accept_pixel = false
            call this%rejected_pixels%set_value(LUDA_pixel_coords,true)
            return accept_pixel;
    }

    coords* yamazaki_test_find_downstream_cell(coords* initial_outlet_pixel) {
        coords* downstream_cell = nullptr;
            downstream_cell => this%cell_neighborhood%yamazaki_find_downstream_cell(initial_outlet_pixel)
        return downstream_cell;
    }

    int* test_generate_cell_cumulative_flow() {
        class(*), dimension(:,:), pointer :: cmltv_flow_array
        call this%generate_cell_cumulative_flow()
        select type(cmltv_flow => this%cell_cumulative_flow)
        type is (latlon_subfield)
            cmltv_flow_array => cmltv_flow%latlon_get_data()
        end select
        return cmltv_flow_array;
    }

    void generate_cell_cumulative_flow() {
        class(doubly_linked_list), pointer :: pixels_to_process
        class(coords), pointer :: upstream_neighbors_coords_list(:)
        class(*), pointer :: pixel_coords
        class(*), allocatable :: neighbor_coords
        bool unprocessed_inflowing_pixels
        int pixel_cumulative_flow
        class(*), pointer :: neighbor_cumulative_flow
        int i
            call this%initialize_cell_cumulative_flow_subfield()
            pixels_to_process => this%generate_list_of_pixels_in_cell()
            do {
                if (pixels_to_process%get_length() <= 0) exit
                do {
                    //Combine iteration and check to see if we have reached the end
                    //of the list
                    if(pixels_to_process%iterate_forward()) exit
                    pixel_cumulative_flow = 1
                    unprocessed_inflowing_pixels = false
                    pixel_coords => pixels_to_process%get_value_at_iterator_position()
                    select type (pixel_coords)
                    class is (coords)
                        upstream_neighbors_coords_list => this%find_upstream_neighbors(pixel_coords)
                    end select
                    if (size(upstream_neighbors_coords_list) > 0) {
                        do i = 1,size(upstream_neighbors_coords_list) {
                            if (allocated(neighbor_coords)) deallocate(neighbor_coords)
                            allocate(neighbor_coords,source=upstream_neighbors_coords_list(i))
                            select type (neighbor_coords)
                            class is (coords)
                                if( .not. this%check_if_coords_are_in_area(neighbor_coords)) {
                                    cycle
                                }
                                neighbor_cumulative_flow => this%cell_cumulative_flow%get_value(neighbor_coords)
                            end select
                            select type (neighbor_cumulative_flow)
                            type is (integer)
                                if (neighbor_cumulative_flow == 0) {
                                    unprocessed_inflowing_pixels = true
                                } else {
                                    pixel_cumulative_flow = pixel_cumulative_flow + neighbor_cumulative_flow
                                }
                            end select
                            deallocate(neighbor_cumulative_flow)
                            if (unprocessed_inflowing_pixels) exit
                        }
                        if (allocated(neighbor_coords)) deallocate(neighbor_coords)
                    }
                    if (.not. unprocessed_inflowing_pixels) {
                        select type (pixel_coords)
                        class is (coords)
                            call this%cell_cumulative_flow%set_value(pixel_coords,pixel_cumulative_flow)
                        end select
                        call pixels_to_process%remove_element_at_iterator_position()
                        if (pixels_to_process%get_length() <= 0) exit
                    }
                    deallocate(upstream_neighbors_coords_list)
                }
                if (associated(upstream_neighbors_coords_list)) deallocate(upstream_neighbors_coords_list)
                call pixels_to_process%reset_iterator()
            }
            deallocate(pixels_to_process)
    }

    void set_contains_river_mouths(bool value) {
        bool value
            this%contains_river_mouths = value
    }

    void init_latlon_field(section_coords* field_section_coords,
                           int* river_directions) {
        class(*), dimension(:,:), pointer :: river_directions
        type(latlon_section_coords) :: field_section_coords
            this%section_min_lat = field_section_coords%section_min_lat
            this%section_min_lon = field_section_coords%section_min_lon
            this%section_width_lat = field_section_coords%section_width_lat
            this%section_width_lon = field_section_coords%section_width_lon
            this%section_max_lat =  field_section_coords%section_min_lat + field_section_coords%section_width_lat - 1
            this%section_max_lon =  field_section_coords%section_min_lon + field_section_coords%section_width_lon - 1
            allocate(this%field_section_coords,source=field_section_coords)
            this%total_cumulative_flow => null()
            this%river_directions => latlon_field_section(river_directions,field_section_coords)
            call this%initialize_cells_to_reprocess_field_section()
    }

    void init_icon_single_index_field(section_coords* field_section_coords,
                                      int* river_directions) {
        class(*), dimension(:), pointer :: river_directions
        type(generic_1d_section_coords) :: field_section_coords
            this%ncells = size(field_section_coords%cell_neighbors,1)
            allocate(this%field_section_coords,source=field_section_coords)
            this%total_cumulative_flow => null()
            this%river_directions => icon_single_index_field_section(river_directions,field_section_coords)
            call this%initialize_cells_to_reprocess_field_section()
    }

    bool* latlon_check_field_for_localized_loops() {
        bool* cells_to_reprocess;
        type(latlon_coords) :: cell_coords
        int i,j
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    cell_coords = latlon_coords(i,j)
                    if (this%check_cell_for_localized_loops(cell_coords)) {
                        call this%cells_to_reprocess%set_value(cell_coords,true)
                    }
                }
            }
            select type (cells_to_reprocess_field_section => this%cells_to_reprocess)
            type is (latlon_field_section)
                cells_to_reprocess_data => cells_to_reprocess_field_section%get_data()
                select type (cells_to_reprocess_data)
                type is (logical)
                    cells_to_reprocess => cells_to_reprocess_data
                end select
            end select
            return cells_to_reprocess;
    }

    bool* icon_single_index_check_field_for_localized_loops() {
        //Retain the dimension from the lat lon version but only ever use a single column for the
        //second index
        bool* cells_to_reprocess
        type(generic_1d_coords) :: cell_coords
        int i
            do i = 1, this%ncells {
                cell_coords = generic_1d_coords(i)
                if (this%check_cell_for_localized_loops(cell_coords)) {
                    call this%cells_to_reprocess%set_value(cell_coords,true)
                }
            }
            allocate(cells_to_reprocess(this%ncells,1))
            select type (cells_to_reprocess_field_section => this%cells_to_reprocess)
            type is (icon_single_index_field_section)
                cells_to_reprocess_data => cells_to_reprocess_field_section%get_data()
                select type (cells_to_reprocess_data)
                type is (logical)
                    cells_to_reprocess(:,1) = cells_to_reprocess_data
                end select
            end select
            deallocate(cells_to_reprocess_data)
            return cells_to_reprocess;
    }

    void latlon_initialize_cells_to_reprocess_field_section() {
        class(*), dimension(:,:), pointer :: cells_to_reprocess_initialization_data
            allocate(logical:: cells_to_reprocess_initialization_data(this%section_width_lat,
                                                                      this%section_width_lon))
            select type(cells_to_reprocess_initialization_data)
            type is (logical)
                cells_to_reprocess_initialization_data = false
            end select
            select type (field_section_coords => this%field_section_coords)
            type is (latlon_section_coords)
                this%cells_to_reprocess => latlon_field_section(cells_to_reprocess_initialization_data,
                                                                field_section_coords)
            end select
    }

    void icon_single_index_initialize_cells_to_reprocess_field_section() {
        class(*), dimension(:), pointer :: cells_to_reprocess_initialization_data
            allocate(logical:: cells_to_reprocess_initialization_data(this%ncells))
            select type(cells_to_reprocess_initialization_data)
            type is (logical)
                cells_to_reprocess_initialization_data = false
            end select
            select type (field_section_coords => this%field_section_coords)
            type is (generic_1d_section_coords)
                this%cells_to_reprocess =>
                    icon_single_index_field_section(cells_to_reprocess_initialization_data,
                                                    field_section_coords)
            end select
    }

    void init_latlon_cell(section_coords* cell_section_coords,
                          int* river_directions,
                          int* total_cumulative_flow) {
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
            allocate(this%cell_section_coords,source=cell_section_coords)
            this%section_min_lat = cell_section_coords%section_min_lat
            this%section_min_lon = cell_section_coords%section_min_lon
            this%section_width_lat = cell_section_coords%section_width_lat
            this%section_width_lon = cell_section_coords%section_width_lon
            this%section_max_lat =  cell_section_coords%section_min_lat + cell_section_coords%section_width_lat - 1
            this%section_max_lon =  cell_section_coords%section_min_lon + cell_section_coords%section_width_lon - 1
            river_directions_pointer => river_directions
            total_cumulative_flow_pointer => total_cumulative_flow
            this%river_directions => latlon_field_section(river_directions_pointer,cell_section_coords)
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,cell_section_coords)
            //Minus four to avoid double counting the corners
            this%number_of_edge_pixels = 2*this%section_width_lat + 2*this%section_width_lon - 4
            call this%initialize_rejected_pixels_subfield()
            this%contains_river_mouths = false
            this%ocean_cell = false
    }

    void init_irregular_latlon_cell(section_coords* cell_section_coords,
                                    int* river_directions,
                                    int* total_cumulative_flow) {
        class(irregular_latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
        class(*), dimension(:,:), pointer :: cell_numbers_pointer
            allocate(this%cell_section_coords,source=cell_section_coords)
            this%section_min_lat = cell_section_coords%section_min_lat
            this%section_min_lon = cell_section_coords%section_min_lon
            this%section_width_lat = cell_section_coords%section_width_lat
            this%section_width_lon = cell_section_coords%section_width_lon
            this%section_max_lat =  cell_section_coords%section_min_lat + cell_section_coords%section_width_lat - 1
            this%section_max_lon =  cell_section_coords%section_min_lon + cell_section_coords%section_width_lon - 1
            cell_numbers_pointer => cell_section_coords%cell_numbers
            this%cell_numbers => latlon_field_section(cell_numbers_pointer,cell_section_coords)
            this%cell_number = cell_section_coords%cell_number
            river_directions_pointer => river_directions
            total_cumulative_flow_pointer => total_cumulative_flow
            this%river_directions => latlon_field_section(river_directions_pointer,cell_section_coords)
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,cell_section_coords)
            this%number_of_edge_pixels = 0
            call this%initialize_rejected_pixels_subfield()
            this%contains_river_mouths = false
            this%ocean_cell = false
    }

    void irregular_latlon_cell_destructor() {
            call cell_destructor(this)
            if(associated(this%cell_numbers)) deallocate(this%cell_numbers)
            call this%cell_neighborhood%destructor()
    }

    void yamazaki_init_latlon_cell(coords* cell_section_coords,
                                   int* river_directions,
                                   int* total_cumulative_flow,
                                   int* yamazaki_outlet_pixels) {
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), target :: yamazaki_outlet_pixels
        class(*), dimension(:,:), pointer :: yamazaki_outlet_pixels_pointer
            call this%init_latlon_cell(cell_section_coords,river_directions,
                                       total_cumulative_flow)
            yamazaki_outlet_pixels_pointer => yamazaki_outlet_pixels
            this%yamazaki_outlet_pixels => latlon_field_section(yamazaki_outlet_pixels_pointer,
                                                                cell_section_coords)
    }

    vector<coords*>* latlon_find_edge_pixels(){
        int i
        int list_index
        list_index = 1
        allocate(latlon_coords::edge_pixel_coords_list(this%number_of_edge_pixels))
        select type (edge_pixel_coords_list)
        type is (latlon_coords)
            do i = this%section_min_lat,this%section_max_lat {
                edge_pixel_coords_list(list_index) = latlon_coords(i,this%section_min_lon)
                edge_pixel_coords_list(list_index+1) = latlon_coords(i,this%section_max_lon)
                list_index = list_index + 2
            }
            do  i = this%section_min_lon+1,this%section_max_lon-1 {
                edge_pixel_coords_list(list_index) = latlon_coords(this%section_min_lat,i)
                edge_pixel_coords_list(list_index+1) = latlon_coords(this%section_max_lat,i)
                list_index = list_index + 2
            }
        end select
        return edge_pixel_coords_list;
    }

    vector<coords*>* irregular_latlon_find_edge_pixels() {
        vector<coords*>* edge_pixel_coords_list;
        type(doubly_linked_list), pointer :: list_of_edge_pixels
        class(*), pointer :: working_pixel_ptr
        type(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        int i,j
        int list_index
        int imax,jmax
        bool is_edge
            cell_numbers => this%cell_numbers
            list_index = 1
            allocate(list_of_edge_pixels)
            select type(cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            this%number_of_edge_pixels = 0
            imax = size(cell_numbers_data,1)
            jmax = size(cell_numbers_data,2)
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    if (cell_numbers_data(i,j) == this%cell_number) {
                        is_edge = false
                        if (i /= 1 .and. j /= 1) {
                            is_edge = (cell_numbers_data(i-1,j-1) /= this%cell_number)
                        }
                        if (i /= 1) {
                            is_edge = (cell_numbers_data(i-1,j)   /= this%cell_number) .or. is_edge
                        }
                        if (i /= 1 .and. j /= jmax) {
                            is_edge = (cell_numbers_data(i-1,j+1) /= this%cell_number) .or. is_edge
                        }
                        if (j /= 1 ) {
                            is_edge = (cell_numbers_data(i,j-1)   /= this%cell_number) .or. is_edge
                        }
                        if (j /= jmax) {
                            is_edge = (cell_numbers_data(i,j+1)   /= this%cell_number) .or. is_edge
                        }
                        if (i /= imax .and. j /= 1) {
                            is_edge = (cell_numbers_data(i+1,j-1) /= this%cell_number) .or. is_edge
                        }
                        if (i /= imax) {
                            is_edge = (cell_numbers_data(i+1,j)   /= this%cell_number) .or. is_edge
                        }
                        if (i /= imax .and. j /= jmax) {
                            is_edge = (cell_numbers_data(i+1,j+1) /= this%cell_number) .or. is_edge
                        }
                        if (is_edge) {
                            call list_of_edge_pixels%add_value_to_back(latlon_coords(i,j))
                            this%number_of_edge_pixels = this%number_of_edge_pixels + 1
                        }
                    }
                }
            }
            allocate(latlon_coords::edge_pixel_coords_list(this%number_of_edge_pixels))
            call list_of_edge_pixels%reset_iterator()
            do while (.not. list_of_edge_pixels%iterate_forward()) {
                working_pixel_ptr => list_of_edge_pixels%get_value_at_iterator_position()
                select type(working_pixel_ptr)
                type is (latlon_coords)
                    select type(edge_pixel_coords_list)
                    type is (latlon_coords)
                        edge_pixel_coords_list(list_index) = working_pixel_ptr
                    end select
                end select
                list_index = list_index + 1
                call list_of_edge_pixels%remove_element_at_iterator_position()
            }
            deallocate(list_of_edge_pixels)
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    double latlon_calculate_length_through_pixel(coords* coords_in) {
        double length_through_pixel;
        if (this%is_diagonal(coords_in)) {
            length_through_pixel = 1.414
        } else {
            length_through_pixel = 1.0
        }
        return length_through_pixel;
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    bool latlon_check_if_coords_are_in_area(coords* coords_in) {
        bool within_area;
        select type (coords_in)
        type is (latlon_coords)
            if (coords_in%lat >= this%section_min_lat .and.
                coords_in%lat <= this%section_max_lat .and.
                coords_in%lon >= this%section_min_lon .and.
                coords_in%lon <= this%section_max_lon) {
                within_area = true
            } else {
                within_area = false
            }
        end select
        return within_area;
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    bool generic_check_if_coords_are_in_area(coords* coords_in) {
        class(coords), intent(in) :: coords_in
        class(field_section), pointer :: cell_numbers
        type(doubly_linked_list), pointer :: list_of_cells_in_neighborhood
        class (*), pointer :: cell_number_ptr
        class (*), pointer :: working_cell_number_ptr
        integer    :: working_cell_number
        bool within_area
        int cell_number
            select type(this)
            class is (irregular_latlon_cell)
                cell_numbers => this%cell_numbers
                cell_number = this%cell_number
            class is (irregular_latlon_neighborhood)
                list_of_cells_in_neighborhood => this%list_of_cells_in_neighborhood
                cell_numbers => this%cell_numbers
            end select
            within_area = false
            select type(this)
            class is (field)
                within_area = true
            class is (cell)
                working_cell_number_ptr => cell_numbers%get_value(coords_in)
                select type (working_cell_number_ptr)
                    type is (integer)
                        working_cell_number = working_cell_number_ptr
                end select
                deallocate(working_cell_number_ptr)
                if(working_cell_number == cell_number) {
                    within_area = true
                }
            class is (neighborhood)
                call list_of_cells_in_neighborhood%reset_iterator()
                do while (.not. list_of_cells_in_neighborhood%iterate_forward()) {
                    cell_number_ptr =>
                        list_of_cells_in_neighborhood%get_value_at_iterator_position()
                    select type (cell_number_ptr)
                    type is (integer)
                        working_cell_number_ptr => cell_numbers%get_value(coords_in)
                        select type (working_cell_number_ptr)
                        type is (integer)
                            working_cell_number = working_cell_number_ptr
                        end select
                        deallocate(working_cell_number_ptr)
                        if(working_cell_number == cell_number_ptr) {
                            within_area = true
                        }
                    end select
                }
            end select
            return within_area;
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    vector<coords*>* latlon_find_upstream_neighbors(coords* coords_in) {
        vector<coords*>* list_of_neighbors = new vector<coords*>();
        int i,j
        int counter
        int n
        counter = 0
        n = 0
        select type (coords_in)
        type is (latlon_coords)
            do i = coords_in%lat + 1, coords_in%lat - 1,-1 {
                do j = coords_in%lon - 1, coords_in%lon + 1 {
                    counter = counter + 1
                    if (counter == 5) cycle
                    if (this%neighbor_flows_to_pixel(latlon_coords(lat=i,lon=j),
                        dir_based_direction_indicator(counter))) {
                        n = n + 1
                        list_of_neighbors_temp(n) = latlon_coords(lat=i,lon=j)
                    }
                }
            }
        end select
        allocate(latlon_coords::list_of_neighbors(n))
        select type(list_of_neighbors)
        type is (latlon_coords)
            do counter = 1,n {
                list_of_neighbors(counter) = list_of_neighbors_temp(counter)
            }
        end select
        return list_of_neighbors;
    }

    void latlon_initialize_cell_cumulative_flow_subfield() {
        class(*), dimension(:,:), pointer :: input_data
        bool wrap
            allocate(integer::input_data(this%section_width_lat,this%section_width_lon))
            select type(input_data)
            type is (integer)
                input_data = 0
            end select
            select type (river_directions => this%river_directions)
            class is (latlon_field_section)
                wrap = (this%section_width_lon == river_directions%get_nlon() .and.
                        river_directions%get_wrap())
            end select
            select type(cell_section_coords => this%cell_section_coords)
            class is (latlon_section_coords)
                this%cell_cumulative_flow => latlon_subfield(input_data,cell_section_coords,
                                                             wrap)
            end select
    }

    void latlon_initialize_rejected_pixels_subfield() {
        class(*), dimension(:,:), pointer :: rejected_pixel_initialization_data
        bool wrap
            allocate(logical::rejected_pixel_initialization_data(this%section_width_lat,
                                                                 this%section_width_lon))
            select type(rejected_pixel_initialization_data)
            type is (logical)
                rejected_pixel_initialization_data = false
            end select
            select type (river_directions => this%river_directions)
            class is (latlon_field_section)
                wrap = (this%section_width_lon == river_directions%get_nlon() .and.
                        river_directions%get_wrap())
            end select
            select type (cell_section_coords => this%cell_section_coords)
            class is (latlon_section_coords)
                this%rejected_pixels => latlon_subfield(rejected_pixel_initialization_data,cell_section_coords,wrap)
            end select
    }

    vector<coords*>* latlon_generate_list_of_pixels_in_cell() {
        vector<coords*>* list_of_pixels_in_cell = new vector<coords*>()
        int i,j
            allocate(list_of_pixels_in_cell)
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    call list_of_pixels_in_cell%add_value_to_back(latlon_coords(i,j))
                }
            }
        return list_of_pixels_in_cell
    }

    vector<coords*>* irregular_latlon_generate_list_of_pixels_in_cell() {
        vector<coords*>*: list_of_pixels_in_cell = new vector<coords*>()
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        int i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            allocate(list_of_pixels_in_cell)
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    if (cell_numbers_data(i,j) == this%cell_number) {
                        call list_of_pixels_in_cell%add_value_to_back(latlon_coords(i,j))
                    }
                }
            }
    }

    int latlon_get_sink_combined_cumulative_flow(coords* main_sink_coords) {
        int combined_cumulative_flow_to_sinks;
        int greatest_flow_to_single_sink
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_sink
        int highest_outflow_sink_lat,highest_outflow_sink_lon
        int i,j
            highest_outflow_sink_lat = 0
            highest_outflow_sink_lon = 0
            combined_cumulative_flow_to_sinks = 0
            greatest_flow_to_single_sink = 0
            flow_direction_for_sink => this%get_flow_direction_for_sink()
            do j = this%section_min_lon, this%section_min_lon + this%section_width_lon - 1 {
                do i = this%section_min_lat, this%section_min_lat + this%section_width_lat - 1 {
                    current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                    current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                    select type (current_pixel_total_cumulative_flow)
                    type is (integer)
                        if (flow_direction_for_sink%is_equal_to(current_pixel_flow_direction)) {
                            combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow +
                                combined_cumulative_flow_to_sinks
                            if (present(main_sink_coords)) {
                                if (greatest_flow_to_single_sink <
                                    current_pixel_total_cumulative_flow) {
                                    current_pixel_total_cumulative_flow =
                                        greatest_flow_to_single_sink
                                    highest_outflow_sink_lat = i
                                    highest_outflow_sink_lon = j
                                }
                            }
                        }
                        deallocate(current_pixel_flow_direction)
                    end select
                    deallocate(current_pixel_total_cumulative_flow)
                }
            }
            deallocate(flow_direction_for_sink)
            if(present(main_sink_coords)) {
                allocate(main_sink_coords,source=latlon_coords(highest_outflow_sink_lat,
                                                               highest_outflow_sink_lon))
            }
            return combined_cumulative_flow_to_sinks;
    }

    int latlon_get_rmouth_outflow_combined_cumulative_flow(OPT coords* main_outflow_coords) {
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_outflow
        int greatest_flow_to_single_outflow
        int combined_cumulative_rmouth_outflow
        int highest_outflow_rmouth_lat,highest_outflow_rmouth_lon
        int i,j
            highest_outflow_rmouth_lat = 0
            highest_outflow_rmouth_lon = 0
            combined_cumulative_rmouth_outflow = 0
            greatest_flow_to_single_outflow = 0
            flow_direction_for_outflow => this%get_flow_direction_for_outflow()
            do j = this%section_min_lon, this%section_min_lon + this%section_width_lon - 1 {
                do i = this%section_min_lat, this%section_min_lat + this%section_width_lat - 1 {
                    current_pixel_total_cumulative_flow =>
                        this%total_cumulative_flow%get_value(latlon_coords(i,j))
                    current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                    select type (current_pixel_total_cumulative_flow)
                    type is (integer)
                        if (flow_direction_for_outflow%is_equal_to(current_pixel_flow_direction)) {
                            combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow +
                                combined_cumulative_rmouth_outflow
                            if (present(main_outflow_coords)) {
                                if (greatest_flow_to_single_outflow <
                                    current_pixel_total_cumulative_flow) {
                                    current_pixel_total_cumulative_flow =
                                        greatest_flow_to_single_outflow
                                    highest_outflow_rmouth_lat = i
                                    highest_outflow_rmouth_lon = j
                                }
                            }
                        }
                    end select
                    deallocate(current_pixel_total_cumulative_flow)
                    deallocate(current_pixel_flow_direction)
                }
            }
            deallocate(flow_direction_for_outflow)
            if(present(main_outflow_coords)) {
                allocate(main_outflow_coords,source=latlon_coords(highest_outflow_rmouth_lat,
                                                                  highest_outflow_rmouth_lon))
            }
            return combined_cumulative_rmouth_outflow;
    }

   int irregular_latlon_get_sink_combined_cumulative_flow(OPT coords* main_sink_coords) {
        int combined_cumulative_flow_to_sinks
        int greatest_flow_to_single_sink
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_sink
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        int highest_outflow_sink_lat,highest_outflow_sink_lon
        int i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            highest_outflow_sink_lat = 0
            highest_outflow_sink_lon = 0
            combined_cumulative_flow_to_sinks = 0
            greatest_flow_to_single_sink = 0
            flow_direction_for_sink => this%get_flow_direction_for_sink()
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    if (cell_numbers_data(i,j) == this%cell_number) {
                        current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                        current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                        select type (current_pixel_total_cumulative_flow)
                        type is (integer)
                            if (flow_direction_for_sink%is_equal_to(current_pixel_flow_direction)) {
                                combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow +
                                    combined_cumulative_flow_to_sinks
                                if (present(main_sink_coords)) {
                                    if (greatest_flow_to_single_sink <
                                        current_pixel_total_cumulative_flow) {
                                        current_pixel_total_cumulative_flow =
                                            greatest_flow_to_single_sink
                                        highest_outflow_sink_lat = i
                                        highest_outflow_sink_lon = j
                                    }
                                }
                            }
                            deallocate(current_pixel_flow_direction)
                        end select
                        deallocate(current_pixel_total_cumulative_flow)
                    }
                }
            }
            deallocate(flow_direction_for_sink)
            if(present(main_sink_coords)) {
                allocate(main_sink_coords,source=latlon_coords(highest_outflow_sink_lat,
                                                               highest_outflow_sink_lon))
            }
            return combined_cumulative_flow_to_sinks;
    }

    int irregular_latlon_get_rmouth_outflow_combined_cumulative_flow(OPT coords* main_outflow_coords){
        class(*), pointer :: current_pixel_total_cumulative_flow
        class(*), pointer :: current_pixel_flow_direction
        class(direction_indicator), pointer :: flow_direction_for_outflow
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        int greatest_flow_to_single_outflow
        int combined_cumulative_rmouth_outflow
        int highest_outflow_rmouth_lat,highest_outflow_rmouth_lon
        int i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            highest_outflow_rmouth_lat = 0
            highest_outflow_rmouth_lon = 0
            combined_cumulative_rmouth_outflow = 0
            greatest_flow_to_single_outflow = 0
            flow_direction_for_outflow => this%get_flow_direction_for_outflow()
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    if (cell_numbers_data(i,j) == this%cell_number) {
                        current_pixel_total_cumulative_flow => this%total_cumulative_flow%get_value(latlon_coords(i,j))
                        current_pixel_flow_direction => this%river_directions%get_value(latlon_coords(i,j))
                        select type (current_pixel_total_cumulative_flow)
                        type is (integer)
                            if (flow_direction_for_outflow%is_equal_to(current_pixel_flow_direction)) {
                                combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow +
                                    combined_cumulative_rmouth_outflow
                                if (present(main_outflow_coords)) {
                                    if (greatest_flow_to_single_outflow <
                                        current_pixel_total_cumulative_flow) {
                                        current_pixel_total_cumulative_flow =
                                            greatest_flow_to_single_outflow
                                        highest_outflow_rmouth_lat = i
                                        highest_outflow_rmouth_lon = j
                                    }
                                }
                            }
                        end select
                        deallocate(current_pixel_total_cumulative_flow)
                        deallocate(current_pixel_flow_direction)
                    }
                }
            }
            deallocate(flow_direction_for_outflow)
            if(present(main_outflow_coords)) {
                allocate(main_outflow_coords,source=latlon_coords(highest_outflow_rmouth_lat,
                                                                  highest_outflow_rmouth_lon))
            }
            return combined_cumulative_rmouth_outflow;
    }


    void latlon_print_area(){
    character(len=160) :: line
    class(*), pointer :: value
    int i,j
        write(*,*) 'Area Information'
        write(*,*) 'min lat=', this%section_min_lat
        write(*,*) 'min lon=', this%section_min_lon
        write(*,*) 'max lat=', this%section_max_lat
        write(*,*) 'max lon=', this%section_max_lon
        write(*,*) 'area rdirs'
        select type(rdirs => this%river_directions)
        type is (latlon_field_section)
            do i = this%section_min_lat,this%section_max_lat {
                line = ''
                do j = this%section_min_lon,this%section_max_lon {
                    value => rdirs%get_value(latlon_coords(i,j))
                    select type(value)
                    type is (integer)
                        write(line,'(A,I5)') trim(line), value
                    end select
                    deallocate(value)
                }
                write (*,'(A)') line
            }
        end select
        write(*,*) 'area total cumulative flow'
        select type(total_cumulative_flow => this%total_cumulative_flow)
        type is (latlon_field_section)
            do i = this%section_min_lat,this%section_max_lat {
                line=''
                do j = this%section_min_lon,this%section_max_lon {
                    value => total_cumulative_flow%get_value(latlon_coords(i,j))
                    select type(value)
                    type is (integer)
                        write(line,'(A,I5)') trim(line), value
                    end select
                    deallocate(value)
                }
                write (*,'(A)') line
            }
        end select
    }

    double icon_single_index_field_dummy_real_function(coords* coords_in) {
        double dummy_result
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            dummy_result = 0.0
            stop
        return dummy_result
    }

    void icon_single_index_field_dummy_subroutine() {
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            stop
    }

    vector<coords*>* icon_single_index_field_dummy_coords_pointer_function(coords* coords_in) {
        vector<coords*>* list_of_neighbors(:);
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            allocate(generic_1d_coords::list_of_neighbors(1))
            stop
        return list_of_neighbors;
    }

    function latlon_dir_based_rdirs_cell_constructor(cell_section_coords,river_directions,
                                                    total_cumulative_flow) {result(constructor)
        type(latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            allocate(constructor)
            call constructor%init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,
                                                              total_cumulative_flow)
    }

    function irregular_latlon_dir_based_rdirs_cell_constructor(cell_section_coords,
                                                               river_directions,
                                                               total_cumulative_flow,
                                                               cell_neighbors) {
                                                                result(constructor)
        type(irregular_latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(irregular_latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
            allocate(constructor)
            call constructor%init_irregular_latlon_dir_based_rdirs_cell(cell_section_coords,
                                                                        river_directions,
                                                                        total_cumulative_flow,
                                                                        cell_neighbors)
    }


    function yamazaki_latlon_dir_based_rdirs_cell_constructor(cell_section_coords,river_directions,
                                                              total_cumulative_flow,yamazaki_outlet_pixels,
                                                              yamazaki_section_coords) {
        result(constructor)
        type(latlon_dir_based_rdirs_cell), allocatable :: constructor
        class(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:) :: yamazaki_outlet_pixels
            allocate(constructor)
            call constructor%yamazaki_init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,
                                                                       total_cumulative_flow,yamazaki_outlet_pixels,
                                                                       yamazaki_section_coords)
    }

    void init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,
                                                total_cumulative_flow) {
        type(latlon_section_coords) :: cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            call this%init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
            allocate(this%cell_neighborhood, source=latlon_dir_based_rdirs_neighborhood(cell_section_coords,
                     river_directions, total_cumulative_flow))
    }

    void init_irregular_latlon_dir_based_rdirs_cell(cell_section_coords,river_directions,
                                                          total_cumulative_flow,cell_neighbors){
        type(irregular_latlon_section_coords), intent(in) :: cell_section_coords
        integer, dimension(:,:), intent(in) :: river_directions
        integer, dimension(:,:), intent(in) :: total_cumulative_flow
        integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
            call this%init_irregular_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
            allocate(this%cell_neighborhood,
                     source=irregular_latlon_dir_based_rdirs_neighborhood(cell_section_coords,
                                                                          cell_neighbors,
                                                                          river_directions,
                                                                          total_cumulative_flow))
    }

    void yamazaki_init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,
                                                         total_cumulative_flow,yamazaki_outlet_pixels,
                                                         yamazaki_section_coords) {
        type(latlon_section_coords) :: cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:) :: yamazaki_outlet_pixels
            call this%yamazaki_init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow,
                                                yamazaki_outlet_pixels)
            allocate(this%cell_neighborhood, source=
                     latlon_dir_based_rdirs_neighborhood(cell_section_coords,river_directions,
                     total_cumulative_flow,yamazaki_outlet_pixels,yamazaki_section_coords))
    }

    void yamazaki_latlon_calculate_river_directions_as_indices(int* coarse_river_direction_indices,
                                                               section_coords* required_coarse_section_coords) {
        class(coords), pointer :: destination_cell_coords
        class(coords), pointer :: initial_outlet_pixel
        class(coords), pointer :: initial_cell_coords
        int outlet_pixel_type
        int lat_offset, lon_offset
            if (present(required_coarse_section_coords)) {
                select type (required_coarse_section_coords)
                    type is (latlon_section_coords)
                        lat_offset = required_coarse_section_coords%section_min_lat - 1
                        lon_offset = required_coarse_section_coords%section_min_lon - 1
                end select
            } else {
                lat_offset = 0
                lon_offset = 0
            }
            initial_outlet_pixel => this%yamazaki_retrieve_initial_outlet_pixel(outlet_pixel_type)
            initial_cell_coords => this%cell_neighborhood%yamazaki_get_cell_coords(initial_outlet_pixel)
            if (outlet_pixel_type == this%yamazaki_sink_outlet .or. outlet_pixel_type == this%yamazaki_rmouth_outlet) {
                select type (initial_cell_coords)
                type is (latlon_coords)
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,1) =
                        -abs(outlet_pixel_type)
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,2) =
                        -abs(outlet_pixel_type)
                end select
                deallocate(initial_cell_coords)
                deallocate(initial_outlet_pixel)
                return
            }
            destination_cell_coords => this%cell_neighborhood%yamazaki_find_downstream_cell(initial_outlet_pixel)
            select type (initial_cell_coords)
            type is (latlon_coords)
                select type (destination_cell_coords)
                type is (latlon_coords)
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,1) =
                        destination_cell_coords%lat - lat_offset
                    coarse_river_direction_indices(initial_cell_coords%lat,initial_cell_coords%lon,2) =
                        destination_cell_coords%lon - lon_offset
                end select
            end select
            deallocate(initial_cell_coords)
            deallocate(destination_cell_coords)
            deallocate(initial_outlet_pixel)
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    function latlon_dir_based_rdirs_neighbor_flows_to_pixel(coords_in,
                  flipped_direction_from_neighbor) {result(neighbor_flows_to_pixel)
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        class(*), pointer :: rdir
        logical neighbor_flows_to_pixel
        integer, dimension(9) :: directions_pointing_to_pixel
        directions_pointing_to_pixel = (/9,8,7,6,5,4,3,2,1/)
        select type(flipped_direction_from_neighbor)
        type is (dir_based_direction_indicator)
            rdir => this%river_directions%get_value(coords_in)
            select type(rdir)
            type is (integer)
                if ( rdir ==
                    directions_pointing_to_pixel(flipped_direction_from_neighbor%get_direction())) {
                    neighbor_flows_to_pixel = true
                } else {
                    neighbor_flows_to_pixel = false
                }
            end select
            deallocate(rdir)
        end select
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    function icon_si_index_based_rdirs_neighbor_flows_to_pixel_dummy(coords_in,
                  flipped_direction_from_neighbor) {result(neighbor_flows_to_pixel)
        class(coords), intent(in) :: coords_in
        class(direction_indicator), intent(in) :: flipped_direction_from_neighbor
        bool neighbor_flows_to_pixel
            select type (this)
            class is (icon_single_index_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            select type (flipped_direction_from_neighbor)
            type is (index_based_direction_indicator)
                continue
            end select
            neighbor_flows_to_pixel = true
    }


    function latlon_dir_based_rdirs_get_flow_direction_for_sink(is_for_cell) {
        result(flow_direction_for_sink)
        class(direction_indicator), pointer :: flow_direction_for_sink
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) continue
            allocate(flow_direction_for_sink,source=
                dir_based_direction_indicator(this%flow_direction_for_sink))
    }

    function latlon_dir_based_rdirs_get_flow_direction_for_outflow(is_for_cell) {
        result(flow_direction_for_outflow)
        class(direction_indicator), pointer :: flow_direction_for_outflow
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) continue
            allocate(flow_direction_for_outflow,source=
                dir_based_direction_indicator(this%flow_direction_for_outflow))
    }

    function latlon_dir_based_rdirs_get_flow_direction_for_ocean_point(is_for_cell) {
        result(flow_direction_for_ocean_point)
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) continue
            allocate(flow_direction_for_ocean_point,source=
                dir_based_direction_indicator(this%flow_direction_for_ocean_point))
    }

    function irregular_latlon_dir_based_rdirs_get_flow_direction_for_sink(is_for_cell) {
        result(flow_direction_for_sink)
        class(direction_indicator), pointer :: flow_direction_for_sink
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) {
                if (is_for_cell) {
                    allocate(flow_direction_for_sink,source=
                        index_based_direction_indicator(this%cell_flow_direction_for_sink))
                } else {
                    allocate(flow_direction_for_sink,source=
                        index_based_direction_indicator(this%flow_direction_for_sink))
                }
            } else {
                allocate(flow_direction_for_sink,source=
                    index_based_direction_indicator(this%flow_direction_for_sink))
            }
    }

    function irregular_latlon_dir_based_rdirs_get_flow_direction_for_outflow(is_for_cell) {
        result(flow_direction_for_outflow)
        class(direction_indicator), pointer :: flow_direction_for_outflow
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) {
                if (is_for_cell) {
                    allocate(flow_direction_for_outflow,source=
                        index_based_direction_indicator(this%cell_flow_direction_for_outflow))
                } else {
                    allocate(flow_direction_for_outflow,source=
                        index_based_direction_indicator(this%flow_direction_for_outflow))
                }
            } else {
                allocate(flow_direction_for_outflow,source=
                    index_based_direction_indicator(this%flow_direction_for_outflow))
            }
    }

    function irregular_latlon_dir_based_rdirs_get_flow_direction_for_o_point(is_for_cell) {
        result(flow_direction_for_ocean_point)
        class(direction_indicator), pointer :: flow_direction_for_ocean_point
        logical, optional, intent(in) :: is_for_cell
            if (present(is_for_cell)) {
                if (is_for_cell) {
                    allocate(flow_direction_for_ocean_point,source=
                        index_based_direction_indicator(this%cell_flow_direction_for_ocean_point))
                } else {
                    allocate(flow_direction_for_ocean_point,source=
                        index_based_direction_indicator(this%flow_direction_for_ocean_point))
                }
            } else {
                allocate(flow_direction_for_ocean_point,source=
                    index_based_direction_indicator(this%flow_direction_for_ocean_point))
            }
    }

    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by latlon_cell and latlon_neighborhood
    function dir_based_rdirs_is_diagonal(coords_in) {result(diagonal)
        class(coords), intent(in) :: coords_in
        class(*), pointer :: rdir
        bool diagonal
            rdir => this%river_directions%get_value(coords_in)
            select type (rdir)
                type is (integer)
                if (rdir == 7 .or. rdir == 9 .or. rdir == 1 .or. rdir == 3) {
                    diagonal = true
                } else {
                    diagonal = false
                }
            end select
            deallocate(rdir)
    }


    // Note must explicit pass this function the object this as it has the nopass attribute
    // so that it can be shared by icon_single_index_cell and icon_single_index_neighborhood
    function icon_si_index_based_rdirs_is_diagonal_dummy(coords_in) {result(diagonal)
        class(coords), intent(in) :: coords_in
        bool diagonal
            select type (this)
            type is (icon_single_index_index_based_rdirs_field)
                continue
            end select
            select type (coords_in)
            type is (generic_1d_coords)
                continue
            end select
            diagonal = false
    }

    void latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points(this) {
        class(*), pointer :: rdir
        bool non_ocean_cell_found
        int i,j
            non_ocean_cell_found = false
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    rdir => this%river_directions%get_value(latlon_coords(i,j))
                    select type (rdir)
                    type is (integer)
                        if( rdir == this%flow_direction_for_outflow ) {
                            this%contains_river_mouths = true
                            this%ocean_cell = false
                        } else if( rdir /= this%flow_direction_for_ocean_point ) {
                            non_ocean_cell_found = true
                        }
                    end select
                    deallocate(rdir)
                    if(this%contains_river_mouths) return
                }
            }
            this%contains_river_mouths = false
            this%ocean_cell = .not. non_ocean_cell_found
    }

    void irregular_latlon_dir_based_rdirs_mark_o_and_rm_points() {
        class(latlon_field_section), pointer :: cell_numbers
        integer, pointer, dimension(:,:) :: cell_numbers_data
        class(*), pointer :: rdir
        bool non_ocean_cell_found
        int i,j
            cell_numbers => this%cell_numbers
            select type (cell_numbers_data_ptr => cell_numbers%data)
            type is (integer)
                cell_numbers_data => cell_numbers_data_ptr
            end select
            non_ocean_cell_found = false
            do j = this%section_min_lon, this%section_max_lon {
                do i = this%section_min_lat, this%section_max_lat {
                    if (cell_numbers_data(i,j) == this%cell_number) {
                        rdir => this%river_directions%get_value(latlon_coords(i,j))
                        select type (rdir)
                        type is (integer)
                            if( rdir == this%flow_direction_for_outflow ) {
                                this%contains_river_mouths = true
                                this%ocean_cell = false
                            } else if( rdir /= this%flow_direction_for_ocean_point ) {
                                non_ocean_cell_found = true
                            }
                        end select
                        deallocate(rdir)
                        if(this%contains_river_mouths) return
                    }
                }
            }
            this%contains_river_mouths = false
            this%ocean_cell = .not. non_ocean_cell_found
    }

    void neighborhood_destructor(this) {
            if (associated(this%river_directions)) deallocate(this%river_directions)
            if (associated(this%total_cumulative_flow)) deallocate(this%total_cumulative_flow)
            if (associated(this%yamazaki_outlet_pixels)) deallocate(this%yamazaki_outlet_pixels)
    }

    function find_downstream_cell(coords* initial_outlet_pixel) {
        class(coords), allocatable :: working_pixel
        int downstream_cell
        class(*), pointer :: initial_cumulative_flow
        class(*), pointer :: last_cell_cumulative_flow
        bool coords_not_in_neighborhood
        bool outflow_pixel_reached
            initial_cumulative_flow => this%total_cumulative_flow%get_value(initial_outlet_pixel)
            coords_not_in_neighborhood = false
            allocate(working_pixel,source=initial_outlet_pixel)
            call this%find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
            do {
                downstream_cell = this%in_which_cell(working_pixel)
                do {
                    last_cell_cumulative_flow => this%total_cumulative_flow%get_value(working_pixel)
                    call this%find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                    if (coords_not_in_neighborhood .or. .not. downstream_cell == this%in_which_cell(working_pixel) .or.
                        outflow_pixel_reached) exit
                    deallocate(last_cell_cumulative_flow)
                }
                if (coords_not_in_neighborhood .or. outflow_pixel_reached) exit
                select type(last_cell_cumulative_flow)
                type is (integer)
                    select type(initial_cumulative_flow)
                    type is (integer)
                        if ((last_cell_cumulative_flow  - initial_cumulative_flow > area_threshold) .and.
                            ( .not. this%is_center_cell(downstream_cell))) exit
                    end select
                end select
                deallocate(last_cell_cumulative_flow)
            }
            deallocate(last_cell_cumulative_flow)
            deallocate(initial_cumulative_flow)
            deallocate(working_pixel)
            return downstream_cell
    }

    coords* yamazaki_find_downstream_cell(coords* initial_outlet_pixel) {
        coords* cell_coords;
        class(coords), pointer :: new_cell_coords
        class(coords), pointer :: working_pixel
        class(*), pointer :: yamazaki_outlet_pixel_type
        bool coords_not_in_neighborhood
        bool outflow_pixel_reached
        bool downstream_cell_found
        real(double_precision) :: downstream_path_length
        int count_of_cells_passed_through
            downstream_cell_found = false
            allocate(working_pixel,source=initial_outlet_pixel)
            cell_coords => this%yamazaki_get_cell_coords(initial_outlet_pixel)
            downstream_path_length = 0.0
            count_of_cells_passed_through = 0
            do
                call this%find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                downstream_path_length = downstream_path_length + this%calculate_length_through_pixel(working_pixel)
                if (coords_not_in_neighborhood) call this%yamazaki_wrap_coordinates(working_pixel)
                new_cell_coords => this%yamazaki_get_cell_coords(working_pixel)
                if ( .not. cell_coords%are_equal_to(new_cell_coords)) {
                    deallocate(cell_coords)
                    allocate(cell_coords,source=new_cell_coords)
                    count_of_cells_passed_through = count_of_cells_passed_through + 1
                }
                deallocate(new_cell_coords)
                yamazaki_outlet_pixel_type => this%yamazaki_outlet_pixels%get_value(working_pixel)
                select type (yamazaki_outlet_pixel_type)
                type is (integer)
                    if (yamazaki_outlet_pixel_type == this%yamazaki_normal_outlet .and.
                        downstream_path_length > MUFP) downstream_cell_found = true
                    if (yamazaki_outlet_pixel_type == this%yamazaki_rmouth_outlet .or.
                        yamazaki_outlet_pixel_type == this%yamazaki_sink_outlet)
                        downstream_cell_found = true
                end select
                deallocate(yamazaki_outlet_pixel_type)
                if (downstream_cell_found) exit
                if (count_of_cells_passed_through > yamazaki_max_range) {
                    deallocate(working_pixel)
                    allocate(working_pixel,source=initial_outlet_pixel)
                    call this%find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
                    if (coords_not_in_neighborhood) call this%yamazaki_wrap_coordinates(working_pixel)
                    deallocate(cell_coords)
                    cell_coords => this%yamazaki_get_cell_coords(working_pixel)
                    exit
                }
            }
            deallocate(working_pixel)
            return cell_coords;
    }

    bool path_reenters_region(coords* coords_inout) {
        bool coords_not_in_neighborhood
        do {
            call this%find_next_pixel_upstream(coords_inout,coords_not_in_neighborhood)
            if (coords_not_in_neighborhood) {
                path_reenters_region = false
                exit
            } else if (this%in_which_cell(coords_inout) == this%center_cell_id) {
                path_reenters_region = true
                exit
            }
        }
        return path_reenters_region;
    }

    bool is_center_cell(int cell_id) {
        bool is_center_cell;
            if (cell_id == this%center_cell_id) {
                is_center_cell = true
            } else {
                is_center_cell = false
            }
        return is_center_cell;
    }

    void init_latlon_neighborhood(center_cell_section_coords,river_directions,
                                        total_cumulative_flow) {
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: section_coords
        integer, target, dimension(:,:) :: river_directions
        integer, target, dimension(:,:) :: total_cumulative_flow
        class (*), pointer, dimension(:,:) :: river_directions_pointer
        class (*), pointer, dimension(:,:) :: total_cumulative_flow_pointer
            this%center_cell_min_lat = center_cell_section_coords%section_min_lat
            this%center_cell_min_lon = center_cell_section_coords%section_min_lon
            this%center_cell_max_lat =  center_cell_section_coords%section_min_lat +
                                        center_cell_section_coords%section_width_lat - 1
            this%center_cell_max_lon =  center_cell_section_coords%section_min_lon +
                                        center_cell_section_coords%section_width_lon - 1
            this%section_min_lat = center_cell_section_coords%section_min_lat -
                                   center_cell_section_coords%section_width_lat
            this%section_min_lon = center_cell_section_coords%section_min_lon -
                                   center_cell_section_coords%section_width_lon
            this%section_max_lat = this%center_cell_max_lat +
                                   center_cell_section_coords%section_width_lat
            this%section_max_lon = this%center_cell_max_lon +
                                   center_cell_section_coords%section_width_lon
            section_coords = latlon_section_coords(this%section_min_lat,this%section_min_lon,
                                                   this%section_max_lat,this%section_max_lon)
            river_directions_pointer => river_directions
            this%river_directions => latlon_field_section(river_directions_pointer,
                                                          section_coords)
            total_cumulative_flow_pointer => total_cumulative_flow
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,
                                                               section_coords)
            this%center_cell_id = 5
    }

    void init_irregular_latlon_neighborhood(center_cell_section_coords,
                                                  coarse_cell_neighbors,
                                                  river_directions,
                                                  total_cumulative_flow) {
        class(irregular_latlon_section_coords) :: center_cell_section_coords
        type(irregular_latlon_section_coords) :: neighborhood_section_coords
        integer, dimension(:,:), target :: river_directions
        integer, dimension(:,:), target :: total_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_cell_neighbors
        integer, dimension(:), pointer :: list_of_cell_numbers
        class(*), dimension(:,:), pointer :: river_directions_pointer
        class(*), dimension(:,:), pointer :: total_cumulative_flow_pointer
        class(*), dimension(:,:), pointer :: cell_numbers_pointer
        class(*), pointer :: cell_number_pointer
        int i
        int max_nneigh_coarse,nneigh_coarse
        int nbr_cell_number
            max_nneigh_coarse = size(coarse_cell_neighbors,2)
            this%center_cell_id = center_cell_section_coords%cell_number
            allocate(this%list_of_cells_in_neighborhood)
            nneigh_coarse = 0
            do i=1,max_nneigh_coarse {
                nbr_cell_number = coarse_cell_neighbors(this%center_cell_id,i)
                if (nbr_cell_number > 0) {
                    call this%list_of_cells_in_neighborhood%
                        add_value_to_back(nbr_cell_number)
                    nneigh_coarse = nneigh_coarse+1
                }
            }
            allocate(list_of_cell_numbers(nneigh_coarse+1))
            list_of_cell_numbers(1) = this%center_cell_id
            call this%list_of_cells_in_neighborhood%reset_iterator()
            i = 1
            do while (.not. this%list_of_cells_in_neighborhood%iterate_forward()) {
                i = i + 1
                cell_number_pointer => this%list_of_cells_in_neighborhood%get_value_at_iterator_position()
                select type (cell_number_pointer)
                type is (integer)
                    list_of_cell_numbers(i) = cell_number_pointer
                end select
            }
            neighborhood_section_coords =
                irregular_latlon_section_coords(list_of_cell_numbers(i),
                                                center_cell_section_coords%cell_numbers,
                                                center_cell_section_coords%section_min_lats,
                                                center_cell_section_coords%section_min_lons,
                                                center_cell_section_coords%section_max_lats,
                                                center_cell_section_coords%section_max_lons )
            cell_numbers_pointer => center_cell_section_coords%cell_numbers
            this%cell_numbers => latlon_field_section(cell_numbers_pointer,neighborhood_section_coords)
            river_directions_pointer => river_directions
            this%river_directions => latlon_field_section(river_directions_pointer,
                                                          neighborhood_section_coords)
            total_cumulative_flow_pointer => total_cumulative_flow
            this%total_cumulative_flow => latlon_field_section(total_cumulative_flow_pointer,
                                                               neighborhood_section_coords)
            call neighborhood_section_coords%irregular_latlon_section_coords_destructor()
            deallocate(list_of_cell_numbers)
    }

    void irregular_latlon_neighborhood_destructor() {
            if (associated(this%list_of_cells_in_neighborhood)) {
                call this%list_of_cells_in_neighborhood%reset_iterator()
                do while ( .not. this%list_of_cells_in_neighborhood%iterate_forward()) {
                    call this%list_of_cells_in_neighborhood%remove_element_at_iterator_position()
                }
                deallocate(this%list_of_cells_in_neighborhood)
            }
            if (associated(this%cell_numbers)) deallocate(this%cell_numbers)
    }

    void yamazaki_init_latlon_neighborhood(center_cell_section_coords,river_directions,
                                                 total_cumulative_flow,yamazaki_outlet_pixels,
                                                 yamazaki_section_coords) {
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), target :: yamazaki_outlet_pixels
        class(*), dimension(:,:), pointer :: yamazaki_outlet_pixels_pointer
            call this%init_latlon_neighborhood(center_cell_section_coords,river_directions,
                                               total_cumulative_flow)
            yamazaki_outlet_pixels_pointer => yamazaki_outlet_pixels
            this%section_min_lat = yamazaki_section_coords%section_min_lat
            this%section_min_lon = yamazaki_section_coords%section_min_lon
            this%section_max_lat = yamazaki_section_coords%section_min_lat + yamazaki_section_coords%section_width_lat - 1
            this%section_max_lon = yamazaki_section_coords%section_min_lon + yamazaki_section_coords%section_width_lon - 1
            this%yamazaki_outlet_pixels => latlon_field_section(yamazaki_outlet_pixels_pointer,
                                                                yamazaki_section_coords)
            this%yamazaki_cell_width_lat = this%center_cell_max_lat + 1 - this%center_cell_min_lat
            this%yamazaki_cell_width_lon = this%center_cell_max_lon + 1 - this%center_cell_min_lon
    }

    function latlon_in_which_cell(coords* pixel) {
        int in_cell;
        int internal_cell_lat
        int internal_cell_lon
            select type(pixel)
            class is (latlon_coords)
                if (pixel%lat < this%center_cell_min_lat) {
                    internal_cell_lat = 1
                } else if (pixel%lat > this%center_cell_max_lat) {
                    internal_cell_lat = 3
                } else {
                    internal_cell_lat = 2
                }
                if (pixel%lon < this%center_cell_min_lon) {
                    internal_cell_lon = 1
                } else if (pixel%lon > this%center_cell_max_lon) {
                    internal_cell_lon = 3
                } else {
                    internal_cell_lon = 2
                }
                in_cell = internal_cell_lon + 9 - 3*internal_cell_lat
            end select
        return in_cell;
    }

    int irregular_latlon_in_which_cell(coords* pixel) {
        class(*), pointer :: cell_number_ptr
        int in_cell
            cell_number_ptr => this%cell_numbers%get_value(pixel)
            select type (cell_number_ptr )
            type is (integer)
                in_cell = cell_number_ptr
            end select
            deallocate(cell_number_ptr)
        return in_cell;
    }

    function yamazaki_latlon_get_cell_coords(pixel_coords) {result(cell_coords)
        class(coords), pointer, intent(in) :: pixel_coords
        class(coords), pointer :: cell_coords
            select type (pixel_coords)
            class is (latlon_coords)
                allocate(cell_coords,source=latlon_coords(ceiling(real(pixel_coords%lat)/this%yamazaki_cell_width_lat),
                                                          ceiling(real(pixel_coords%lon)/this%yamazaki_cell_width_lon)))
            end select
    }


    function yamazaki_irregular_latlon_get_cell_coords_dummy(pixel_coords) {
            result(cell_coords)
        class(coords), pointer, intent(in) :: pixel_coords
        class(coords), pointer :: cell_coords
            //Prevents compiler warnings
            select type (pixel_coords)
            class default
                continue
            end select
            select type (this)
            class default
                continue
            end select
            cell_coords => null()
    }

    //This will neeed updating if we use neighborhood smaller than the full grid size for the
    //yamazaki-style algorithm - at the moment it assumes at neighborhood edges are edges of the
    //full latitude-longitude grid and need longitudinal wrapping.
    void yamazaki_latlon_wrap_coordinates(pixel_coords){
        class(coords), intent(inout) :: pixel_coords
            if ( .not. yamazaki_wrap ) return
            select type (pixel_coords)
            class is (latlon_coords)
                if (pixel_coords%lon < this%section_min_lon) {
                    pixel_coords%lon = this%section_max_lon + pixel_coords%lon + 1 - this%section_min_lon
                } else if (pixel_coords%lon > this%section_max_lon) {
                    pixel_coords%lon = this%section_min_lon + pixel_coords%lon - 1 - this%section_max_lon
                }
            end select
    }

    void yamazaki_irregular_latlon_wrap_coordinates_dummy(pixel_coords){
        class(coords), intent(inout) :: pixel_coords
            //Prevents compiler warnings
            select type (pixel_coords)
            class default
                continue
            end select
            select type (this)
            class default
                continue
            end select
    }

    function latlon_dir_based_rdirs_field_constructor(field_section_coords,river_directions) {
        result(constructor)
        type(latlon_dir_based_rdirs_field), allocatable :: constructor
        type(latlon_section_coords) :: field_section_coords
        integer, dimension(:,:), target :: river_directions
        class(*), dimension(:,:), pointer :: river_directions_pointer
            allocate(constructor)
            river_directions_pointer => river_directions
            call constructor%init_latlon_field(field_section_coords,river_directions_pointer)
    }

    function latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,river_directions,
                                                               total_cumulative_flow) {result(constructor)
        type(latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(latlon_section_coords) :: center_cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
            allocate(constructor)
            call constructor%init_latlon_neighborhood(center_cell_section_coords,river_directions,
                                                      total_cumulative_flow)
    }

    function irregular_latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,
                                                                       coarse_cell_neighbors,
                                                                       river_directions,
                                                                       total_cumulative_flow) {
                                                                        result(constructor)
        type(irregular_latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(irregular_latlon_section_coords) :: center_cell_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:), pointer :: coarse_celL_neighbors
            allocate(constructor)
            call constructor%init_irregular_latlon_neighborhood(center_cell_section_coords,
                                                                coarse_cell_neighbors,
                                                                river_directions,
                                                                total_cumulative_flow)
    }

    function icon_single_index_index_based_rdirs_field_constructor(field_section_coords,river_directions)
            result(constructor)
        type(icon_single_index_index_based_rdirs_field), allocatable :: constructor
        type(generic_1d_section_coords) :: field_section_coords
        integer, dimension(:), target :: river_directions
        class(*), dimension(:), pointer :: river_directions_pointer
            allocate(constructor)
            river_directions_pointer => river_directions
            call constructor%init_icon_single_index_field(field_section_coords,river_directions_pointer)
    }

    function yamazaki_latlon_dir_based_rdirs_neighborhood_constructor(center_cell_section_coords,river_directions,
                                                                      total_cumulative_flow,yamazaki_outlet_pixels,
                                                                      yamazaki_section_coords) {
        result(constructor)
        type(latlon_dir_based_rdirs_neighborhood), allocatable :: constructor
        type(latlon_section_coords) :: center_cell_section_coords
        type(latlon_section_coords) :: yamazaki_section_coords
        integer, dimension(:,:) :: river_directions
        integer, dimension(:,:) :: total_cumulative_flow
        integer, dimension(:,:) :: yamazaki_outlet_pixels
            allocate(constructor)
            call constructor%yamazaki_init_latlon_neighborhood(center_cell_section_coords,river_directions,
                                                               total_cumulative_flow,yamazaki_outlet_pixels,
                                                               yamazaki_section_coords)
    }

    void latlon_dir_based_rdirs_find_next_pixel_downstream(coords_inout,coords_not_in_area,
                                                                 outflow_pixel_reached) {
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
        class(*), pointer :: rdir
        select type (coords_inout)
        type is (latlon_coords)
            rdir => this%river_directions%get_value(coords_inout)
            select type (rdir)
            type is (integer)
                if ( rdir == 7 .or. rdir == 8 .or. rdir == 9) {
                    coords_inout%lat = coords_inout%lat - 1
                } else if ( rdir == 1 .or. rdir == 2 .or. rdir == 3) {
                    coords_inout%lat = coords_inout%lat + 1
                }
                if ( rdir == 7 .or. rdir == 4 .or. rdir == 1 ) {
                    coords_inout%lon = coords_inout%lon - 1
                } else if ( rdir == 9 .or. rdir == 6 .or. rdir == 3 ) {
                    coords_inout%lon = coords_inout%lon + 1
                }
                if ( rdir == 5 .or. rdir == 0 .or. rdir == -1 ) {
                    coords_not_in_area = false
                    if (present(outflow_pixel_reached)) outflow_pixel_reached =  true
                } else {
                    if (present(outflow_pixel_reached)) outflow_pixel_reached =  false
                    if (this%check_if_coords_are_in_area(coords_inout)) {
                        coords_not_in_area = false
                    } else {
                        coords_not_in_area = true
                    }
                }
            end select
            deallocate(rdir)
        end select
    }

    void icon_single_index_index_based_rdirs_find_next_pixel_downstream(
                                                                              coords_inout,
                                                                              coords_not_in_area,
                                                                              outflow_pixel_reached) {
        class(coords), intent(inout) :: coords_inout
        logical, intent(out) :: coords_not_in_area
        logical, intent(out), optional :: outflow_pixel_reached
        class(*), pointer :: next_cell_index
        select type (coords_inout)
        type is (generic_1d_coords)
            select type (this)
            type is (icon_single_index_index_based_rdirs_field)
                next_cell_index => this%river_directions%get_value(coords_inout)
                select type (next_cell_index)
                type is (integer)
                    if ( next_cell_index == this%index_for_sink .or.
                         next_cell_index == this%index_for_outflow .or.
                         next_cell_index == this%index_for_ocean_point ) {
                        coords_not_in_area = false
                        if (present(outflow_pixel_reached)) outflow_pixel_reached =  true
                    } else {
                        if (present(outflow_pixel_reached)) outflow_pixel_reached =  false
                        if (this%check_if_coords_are_in_area(coords_inout)) {
                            coords_inout%index = next_cell_index
                            coords_not_in_area = false
                        } else {
                            coords_not_in_area = true
                        }
                    }
                end select
            end select
            deallocate(next_cell_index)
        end select
    }

    function dir_based_rdirs_calculate_direction_indicator(downstream_cell) {result(flow_direction)
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
            //Would be better to assign this directly but has not yet been implemented
            //in all Fortran compilers
            allocate(flow_direction,source=dir_based_direction_indicator(downstream_cell))
            //Pointless test to get rid of compiler warning that 'this' is unused
            if(this%section_min_lat /= 0) continue
    }

    function irregular_latlon_calculate_direction_indicator(downstream_cell) {
            result(flow_direction)
        integer, intent(in) :: downstream_cell
        class(direction_indicator), pointer :: flow_direction
            allocate(flow_direction,source=index_based_direction_indicator(downstream_cell))
            //Pointless test to get rid of compiler warning that 'this' is unused
            if(this%section_min_lat /= 0) continue
    }
