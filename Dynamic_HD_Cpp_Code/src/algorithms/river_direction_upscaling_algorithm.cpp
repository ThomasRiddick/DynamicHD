void find_next_pixel_upstream(coords* coords_inout,
                              bool& coords_not_in_area){
    int max_cmltv_flow_working_value = 0;
    vector<coords*> upstream_neighbors_coords_list = find_upstream_neighbors(coords_inout);
    if (! upstream_neighbors_coords_list.empty()) {
        for(vector<coords*>::iterator i = upstream_neighbors_coords_list.begin();
                                      i != upstream_neighbors_coords_list.end(); ++i){
            int upstream_neighbor_total_cumulative_flow =
                (*total_cumulative_flow)(i);
            if ( upstream_neighbor_total_cumulative_flow > max_cmltv_flow_working_value) {
                if (coords_inout) delete coords_inout;
                coords_inout = i->clone();
                max_cmltv_flow_working_value =
                    upstream_neighbor_total_cumulative_flow;
            }
        }
        coords_not_in_area = ! check_if_coords_are_in_area(coords_inout);
    } else {
        coords_not_in_area = true;
    }
    delete upstream_neighbors_coords_list;
}

void field_destructor(){
    if (cells_to_reprocess) cells_to_reprocess->deallocate_data();
    if (river_directions) delete river_directions;
    if (total_cumulative_flow) delete total_cumulative_flow;
    if (cells_to_reprocess) delete cells_to_reprocess;
}

bool check_cell_for_localized_loops(coords* coords_inout){
    coords* initial_cell_coords = coords_inout->clone();
    bool coords_not_in_area_dummy;
    find_next_pixel_downstream(coords_inout,coords_not_in_area_dummy);
    bool is_sea_point_or_horizontal_edge_or_sink;
    if ((*coords_inout)==(*initial_cell_coords)) {
        is_sea_point_or_horizontal_edge_or_sink = true;
    } else {
        is_sea_point_or_horizontal_edge_or_sink = false;
    }
    find_next_pixel_downstream(coords_inout,true);
    if ((*coords_inout)==(*initial_cell_coords) &&
        ! is_sea_point_or_horizontal_edge_or_sink) {
        delete initial_cell_coords;
        return localized_loop_found = true;
    } else {
        delete initial_cell_coords;
        return localized_loop_found = false;
    }
}

coords* find_outlet_pixel(bool& no_remaining_outlets,
                          bool use_LCDA_criterion,
                          bool& outlet_is_LCDA = false){
    coords* LCDA_pixel_coords;
    if (use_LCDA_criterion) {
        generate_cell_cumulative_flow();
        LCDA_pixel_coords = find_pixel_with_LCDA();
    } else {
        LCDA_pixel_coords = nullptr;
    }
    coords* LUDA_pixel_coords = nullptr;
    while (true) {
        if (LUDA_pixel_coords) delete LUDA_pixel_coords;
        LUDA_pixel_coords = find_pixel_with_LUDA(no_remaining_outlets);
        if (no_remaining_outlets) {
            outlet_is_LCDA = false;
            break;
        } else if (check_MUFP(LUDA_pixel_coords,LCDA_pixel_coords,outlet_is_LCDA)) break;
    }
    if (use_LCDA_criterion) delete LCDA_pixel_coords;
    return LUDA_pixel_coords;
}

void check_for_sinks_and_rmouth_outflows(coords* outlet_pixel_coords,
                                         bool no_remaining_outlets,
                                         bool& mark_sink,
                                         bool& mark_outflow){
    mark_sink = false;
    mark_outflow = false;
    int outlet_pixel_cumulative_flow;
    if (no_remaining_outlets) {
        outlet_pixel_cumulative_flow = -1;
    } else {
        outlet_pixel_cumulative_flow =
            (*total_cumulative_flow)(outlet_pixel_coords);
    }
    int cumulative_sink_outflow;
    coords* main_sink_coords;
    if (run_check_for_sinks) {
        cumulative_sink_outflow =
            get_sink_combined_cumulative_flow(main_sink_coords);
    } else {
        cumulative_sink_outflow = 0;
    }
    int cumulative_rmouth_outflow;
    coords* main_river_mouth_coords;
    if (contains_river_mouths) {
        cumulative_rmouth_outflow =
            get_rmouth_outflow_combined_cumulative_flow(main_river_mouth_coords);
    } else {
        cumulative_rmouth_outflow = -1;
    }
    if ( cumulative_sink_outflow > cumulative_rmouth_outflow &&
         cumulative_sink_outflow > outlet_pixel_cumulative_flow ) {
         mark_sink = true;
         delete outlet_pixel_coords;
         if (contains_river_mouths) delete main_river_mouth_coords;
         outlet_pixel_coords =  main_sink_coords;
    } else if (  cumulative_rmouth_outflow > outlet_pixel_cumulative_flow) {
         mark_outflow = true;
         delete outlet_pixel_coords;
         if (run_check_for_sinks) delete main_sink_coords;
         outlet_pixel_coords =  main_river_mouth_coords;
    } else {
        if (contains_river_mouths) delete main_river_mouth_coords;
        if (run_check_for_sinks) delete main_sink_coords;
    }
}


direction_indicator* find_cell_flow_direction(coords* outlet_pixel_coords,
                                              bool& no_remaining_outlets) {
    bool mark_sink;
    bool mark_outflow;
    check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink,
                                        mark_outflow);
    if (mark_sink) {
        return get_flow_direction_for_sink(true);
    } else if (mark_outflow) {
        return get_flow_direction_for_outflow(true);
    }
    coords* downstream_cell =
        cell_neighborhood->find_downstream_cell(outlet_pixel_coords)->clone();
    return cell_neighborhood->calculate_direction_indicator(downstream_cell);
}

direction_indicator* process_cell(){
    direction_indicator* flow_direction;
    if (! ocean_cell) {
        bool use_LCDA_criterion = true;
        bool no_remaining_outlets;
        coords* outlet_pixel_coords = find_outlet_pixel(no_remaining_outlets,
                                                        use_LCDA_criterion)
        flow_direction = find_cell_flow_direction(outlet_pixel_coords,
                                                  no_remaining_outlets)
        delete outlet_pixel_coords;
    } else {
        bool is_for_cell = true;
        flow_direction = get_flow_direction_for_ocean_point(is_for_cell);
    }
    return flow_direction;
}

void yamazaki_mark_cell_outlet_pixel(bool use_LCDA_criterion) {
    if (! ocean_cell) {
        bool outlet_is_LCDA = false;
        bool no_remaining_outlets;
        coords* outlet_pixel_coords = find_outlet_pixel(no_remaining_outlets,use_LCDA_criterion,
                                                        outlet_is_LCDA);
        bool mark_sink;
        bool mark_outflow;
        check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink,
                                            mark_outflow);
        int outlet_type;
        if (mark_sink) {
            outlet_type = yamazaki_sink_outlet;
        } else if (mark_outflow) {
            outlet_type = yamazaki_rmouth_outlet;
        } else {
            if ( ! (no_remaining_outlets || outlet_is_LCDA) ) {
                outlet_type = yamazaki_normal_outlet;
            } else {
                outlet_type = yamazaki_source_cell_outlet;
            }
        }
        (*yamazaki_outlet_pixels)(outlet_pixel_coords) = outlet_type;
        delete outlet_pixel_coords;
    }
}

void print_cell_and_neighborhood(){
    cout << (*,*) "For cell" << endl;
    print_area();
    cout <<  "contains_river_mouths= " << contains_river_mouths << endl;
    cout <<  "ocean_cell= " << ocean_cell << endl:
    cout << "For neighborhood" << endl;
    cell_neighborhood->print_area(cell_neighborhood);
}

void cell_destructor() {
    if (rejected_pixels) {
        rejected_pixels->destructor();
        delete rejected_pixels;
    }
    if (cell_cumulative_flow) {
        cell_cumulative_flow->destructor();
        delete cell_cumulative_flow;
    }
    if (river_directions) delete river_directions;
    if (total_cumulative_flow) delete total_cumulative_flow;
    if (cell_neighborhood) cell_neighborhood->destructor()
    if (yamazaki_outlet_pixels) delete yamazaki_outlet_pixels;
}

coords* yamazaki_retrieve_initial_outlet_pixel(int& outlet_pixel_type) {
    forward_list<coords*>* edge_pixel_coords_list = find_edge_pixels();
    bool pixel_found = false;
    coords* initial_outlet_pixel;
    for(forward_list<coords*>::iterator i = edge_pixel_coords_list.begin();
        i != edge_pixel_coords_list.end(); ++i) {
        if ( ! pixel_found ) {
            working_pixel_value = (*yamazaki_outlet_pixels)(i);
                if (working_pixel_value == yamazaki_source_cell_outlet ||
                    working_pixel_value == yamazaki_normal_outlet ||
                    working_pixel_value == yamazaki_sink_outlet ||
                    working_pixel_value == yamazaki_rmouth_outlet) {
                    outlet_pixel_type = working_pixel_value;
                    initial_outlet_pixel = i->clone();
                    pixel_found = true;
                }
            delete working_pixel_value;
        }
    }
    delete edge_pixel_coords_list;
    if ( ! pixel_found ) {
        forward_list<coords*>* pixels_to_process = generate_list_of_pixels_in_cell();
        while(true) {
            //Combine iteration and check to see if we have reached the end
            //of the list
            if (pixels_to_process->empty()) {
                if (pixel_found) {
                    break;
                } else {
                    throw runtime_error("no outlet pixel could "
                                        "be retrieved for cell")
                }
            }
            coords* working_pixel =
                pixels_to_process->front();
            if ( ! pixel_found) {
                working_pixel_value = (*yamazaki_outlet_pixels)(working_pixel);
                    if (working_pixel_value == yamazaki_sink_outlet ||
                        working_pixel_value == yamazaki_rmouth_outlet) {
                        outlet_pixel_type = working_pixel_value;
                        initial_outlet_pixel = working_pixel->clone();
                        pixel_found = true;
                    }
                delete working_pixel_value;
            }
            pixels_to_process->pop_front();
        }
    }
    return initial_outlet_pixel;
}

coords* find_pixel_with_LUDA(bool& no_remaining_outlets) {
    int LUDA_working_value = 0
    coords* LUDA_pixel_coords = nullptr;
    forward_list<coords*>* edge_pixel_coords_list = find_edge_pixels();
    for(forward_list<coords*>::iterator i = edge_pixel_coords_list.begin();
        i != edge_pixel_coords_list.end(); ++i) {
        bool rejected_pixel = (*rejected_pixels)(i);
        if ( ! rejected_pixel) {
            int upstream_drainage = (*total_cumulative_flow)(i);
            if (upstream_drainage > LUDA_working_value) {
                    coords* working_pixel_coords=i->clone()
                    bool coords_not_in_cell;
                    find_next_pixel_downstream(working_pixel_coords,coords_not_in_cell)
                    delete working_pixel_coords;
                    if(coords_not_in_cell) {
                        LUDA_working_value = upstream_drainage
                        if (LUDA_pixel_coords) delete LUDA_pixel_coords;
                        LUDA_pixel_coords=i->clone()
                    }
                }
        }
    }
    if (! LUDA_pixel_coords) {
        LUDA_pixel_coords=edge_pixel_coords_list[1]->clone()
        no_remaining_outlets=true
    } else {
        no_remaining_outlets=false
    }
    delete edge_pixel_coords_list;
    return LUDA_pixel_coords;
}

coords* find_pixel_with_LCDA() {
    int LCDA_pixel_working_value = 0
    coords* LCDA_pixel_coords = nullptr;
    forward_list<coords*>* edge_pixel_coords_list = find_edge_pixels();
    for(forward_list<coords*>::iterator i = edge_pixel_coords_list.begin();
        i != edge_pixel_coords_list.end(); ++i) {
        int cell_drainage_area = (*cell_cumulative_flow)(i);
        if (cell_drainage_area > LCDA_pixel_working_value) {
            coords* working_pixel_coords=i->clone();
            bool coords_not_in_cell;
            find_next_pixel_downstream(working_pixel_coords,
                                       coords_not_in_cell);
            delete working_pixel_coords;
            if(coords_not_in_cell) {
                LCDA_pixel_working_value = cell_drainage_area;
                if (LCDA_pixel_coords) delete LCDA_pixel_coords;
                LCDA_pixel_coords=i->clone();
            }
        }
    }
    if (! LCDA_pixel_coords) {
        LCDA_pixel_coords=edge_pixel_coords_list(1)->clone()
    }
    delete edge_pixel_coords_list;
    return LCDA_pixel_coords;
}

double measure_upstream_path(coords* outlet_pixel_coords) {
    double upstream_path_length = 0.0;
    coords* working_coords = outlet_pixel_coords->clone();
    while(true) {
        double length_through_pixel = calculate_length_through_pixel(working_coords);
        while(true) {
            upstream_path_length = upstream_path_length + length_through_pixel;
            bool coords_not_in_cell;
            find_next_pixel_upstream(working_coords,coords_not_in_cell);
            length_through_pixel = calculate_length_through_pixel(working_coords);
            if (coords_not_in_cell) break;
        }
        if (! cell_neighborhood->path_reenters_region(working_coords)) break;
    }
    delete working_coords;
    return upstream_path_length;
}

bool check_MUFP(coords* LUDA_pixel_coords,
                coords* LCDA_pixel_coords,
                bool& LUDA_is_LCDA = false) {
    LUDA_is_LCDA = false;
    if (measure_upstream_path(LUDA_pixel_coords) > MUFP) {
        return true;
    } else if (LCDA_pixel_coords) {
        if((*LUDA_pixel_coords)==(*LCDA_pixel_coords)) {
            LUDA_is_LCDA = true;
            return true;
        }
    }
    (*rejected_pixels)(LUDA_pixel_coords) = true
    return false;
}

coords* yamazaki_test_find_downstream_cell(coords* initial_outlet_pixel) {
    return cell_neighborhood->yamazaki_find_downstream_cell(initial_outlet_pixel);
}

int* test_generate_cell_cumulative_flow() {
    class(*), dimension(:,:), pointer :: cmltv_flow_array
    generate_cell_cumulative_flow()
    (cmltv_flow = cell_cumulative_flow)
    type is (latlon_subfield)
        cmltv_flow_array = cmltv_flow->latlon_get_data()
    return cmltv_flow_array;
}

void generate_cell_cumulative_flow() {
    initialize_cell_cumulative_flow_subfield();
    vector<coords*>* pixels_to_process = generate_list_of_pixels_in_cell();
    while(! pixels_to_process->empty()) {
        for (vector<coords*>::iterator i = pixels_to_process.begin();
             i != pixels_to_process.end();){
            //Combine iteration and check to see if we have reached the end
            //of the list
            int pixel_cumulative_flow = 1
            bool unprocessed_inflowing_pixels = false
            vector<coords*> upstream_neighbors_coords_list =
                find_upstream_neighbors(i);
            if (! upstream_neighbors_coords_list.empty()) {
                coords* neighbor_coords;
                for(forward_list<coords*>::iterator j = upstream_neighbors_coords_list.begin();
                    j != upstream_neighbors_coords_list.end(); ++j) {
                    coords* neighbor_coords=->clone()
                    if( ! check_if_coords_are_in_area(neighbor_coords)) {
                        delete neighbors_coords;
                        continue;
                    }
                    int neighbor_cumulative_flow = (*cell_cumulative_flow)(neighbor_coords);
                    if (neighbor_cumulative_flow == 0) {
                        unprocessed_inflowing_pixels = true;
                    } else {
                        pixel_cumulative_flow = pixel_cumulative_flow +
                                                neighbor_cumulative_flow;
                    }
                    if (unprocessed_inflowing_pixels) {
                        delete neighbors_coords;
                        break;
                    }
                    delete neighbors_coords;
                }
            }
            delete upstream_neighbors_coords_list;
            if (! unprocessed_inflowing_pixels) {
                    (*cell_cumulative_flow)(i) = pixel_cumulative_flow;
                i = erase_after(--i);
            } else i++;
        }
    }
    delete pixels_to_process;
}

void set_contains_river_mouths(bool value) {
    contains_river_mouths = value;
}

void init_latlon_field(section_coords* field_section_coords,
                       int* river_directions) {
    class(*), dimension(:,:), pointer :: river_directions
    type(latlon_section_coords) :: field_section_coords
    section_min_lat = field_section_coords->section_min_lat
    section_min_lon = field_section_coords->section_min_lon
    section_width_lat = field_section_coords->section_width_lat
    section_width_lon = field_section_coords->section_width_lon
    section_max_lat =  field_section_coords->section_min_lat + field_section_coords->section_width_lat - 1
    section_max_lon =  field_section_coords->section_min_lon + field_section_coords->section_width_lon - 1
    total_cumulative_flow = null()
    river_directions = latlon_field_section(river_directions,field_section_coords)
    initialize_cells_to_reprocess_field_section()
}

void init_icon_single_index_field(section_coords* field_section_coords,
                                  int* river_directions) {
    class(*), dimension(:), pointer :: river_directions
    type(generic_1d_section_coords) :: field_section_coords
    ncells = len(field_section_coords->cell_neighbors,1)
    total_cumulative_flow = null();
    river_directions = icon_single_index_field_section(river_directions,field_section_coords)
    initialize_cells_to_reprocess_field_section()
}

bool* field??? latlon_check_field_for_localized_loops() {
    bool* cells_to_reprocess; field????
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* cell_coords = new latlon_coords(i,j);
            if (check_cell_for_localized_loops(cell_coords)) {
                (*cells_to_reprocess)(cell_coords) = true;
            }
        }
    }
    return cells_to_reprocess;
}

bool* icon_single_index_check_field_for_localized_loops() {
    bool* cells_to_reprocess;
    for (int i = 0; i < ncells; i++) {
        coords* cell_coords = new generic_1d_coords(i)
        if (check_cell_for_localized_loops(cell_coords)) {
            (*cells_to_reprocess)(cell_coords) = true;
        }
    }
    (cells_to_reprocess_field_section = cells_to_reprocess)
    type is (icon_single_index_field_section)
        cells_to_reprocess_data = cells_to_reprocess_field_section->get_data()
            cells_to_reprocess(:,1) = cells_to_reprocess_data
    delete cells_to_reprocess_data;
    return cells_to_reprocess;
}

void latlon_initialize_cells_to_reprocess_field_section() {
    class(*), dimension(:,:), pointer :: cells_to_reprocess_initialization_data
    allocate(logical:: cells_to_reprocess_initialization_data(section_width_lat,
                                                              section_width_lon))
        cells_to_reprocess_initialization_data = false
    (field_section_coords = field_section_coords)
    type is (latlon_section_coords)
        cells_to_reprocess = latlon_field_section(cells_to_reprocess_initialization_data,
                                                        field_section_coords)
}

void icon_single_index_initialize_cells_to_reprocess_field_section() {
    class(*), dimension(:), pointer :: cells_to_reprocess_initialization_data
    allocate(logical:: cells_to_reprocess_initialization_data(ncells))
        cells_to_reprocess_initialization_data = false
    (field_section_coords = field_section_coords)
    type is (generic_1d_section_coords)
        cells_to_reprocess =
            icon_single_index_field_section(cells_to_reprocess_initialization_data,
                                            field_section_coords)
}

void init_latlon_cell(section_coords* cell_section_coords,
                      int* river_directions,
                      int* total_cumulative_flow) {
    type(latlon_section_coords) :: cell_section_coords
    integer, dimension(:,:), target :: river_directions
    integer, dimension(:,:), target :: total_cumulative_flow
    section_min_lat = cell_section_coords->section_min_lat
    section_min_lon = cell_section_coords->section_min_lon
    section_width_lat = cell_section_coords->section_width_lat
    section_width_lon = cell_section_coords->section_width_lon
    section_max_lat =  cell_section_coords->section_min_lat + cell_section_coords->section_width_lat - 1
    section_max_lon =  cell_section_coords->section_min_lon + cell_section_coords->section_width_lon - 1
    river_directions = latlon_field_section(river_directions,cell_section_coords);
    total_cumulative_flow = latlon_field_section(total_cumulative_flow,cell_section_coords);
    //Minus four to avoid double counting the corners
    number_of_edge_pixels = 2*section_width_lat + 2*section_width_lon - 4;
    initialize_rejected_pixels_subfield()
    contains_river_mouths = false;
    ocean_cell = false;
}

void init_irregular_latlon_cell(section_coords* cell_section_coords,
                                int* river_directions,
                                int* total_cumulative_flow) {
    class(irregular_latlon_section_coords) :: cell_section_coords
    integer, dimension(:,:), target :: river_directions
    integer, dimension(:,:), target :: total_cumulative_flow
    class(*), dimension(:,:), pointer :: cell_numbers_pointer
    section_min_lat = cell_section_coords->section_min_lat
    section_min_lon = cell_section_coords->section_min_lon
    section_width_lat = cell_section_coords->section_width_lat
    section_width_lon = cell_section_coords->section_width_lon
    section_max_lat =  cell_section_coords->section_min_lat + cell_section_coords->section_width_lat - 1
    section_max_lon =  cell_section_coords->section_min_lon + cell_section_coords->section_width_lon - 1
    cell_numbers_pointer = cell_section_coords->cell_numbers
    cell_numbers = latlon_field_section(cell_numbers_pointer,cell_section_coords)
    cell_number = cell_section_coords->cell_number
    river_directions = latlon_field_section(river_directions,cell_section_coords)
    total_cumulative_flow = latlon_field_section(total_cumulative_flow,cell_section_coords)
    number_of_edge_pixels = 0
    initialize_rejected_pixels_subfield()
    contains_river_mouths = false
    ocean_cell = false
}

void irregular_latlon_cell_destructor() {
    cell_destructor(this)
    if(cell_numbers) delete cell_numbers;
    cell_neighborhood->destructor()
}

void yamazaki_init_latlon_cell(coords* cell_section_coords,
                               int* river_directions,
                               int* total_cumulative_flow,
                               int* yamazaki_outlet_pixels) {
    type(latlon_section_coords) :: cell_section_coords
    integer, dimension(:,:) :: river_directions
    integer, dimension(:,:) :: total_cumulative_flow
    integer, dimension(:,:), target :: yamazaki_outlet_pixels
    init_latlon_cell(cell_section_coords,river_directions,
                               total_cumulative_flow)
    yamazaki_outlet_pixels = latlon_field_section(yamazaki_outlet_pixels,
                                                        cell_section_coords)
}

forward_list<coords*>* latlon_find_edge_pixels(){
    forward_list<coords*>* edge_pixel_coords_list = new forward_list<coords*>;
    for (int i = section_min_lat; i <= section_max_lat; i++) {
        edge_pixel_coords_list->push_front(new latlon_coords(i,section_min_lon));
        edge_pixel_coords_list->push_front(new latlon_coords(i,section_max_lon));
    }
    for (int i = section_min_lon+1; i <= section_max_lon-1; i++) {
        edge_pixel_coords_list->push_front(latlon_coords(section_min_lat,i));
        edge_pixel_coords_list->push_front(latlon_coords(section_max_lat,i));
    }
    return edge_pixel_coords_list;
}

forward_list<coords*>* irregular_latlon_find_edge_pixels() {
    forward_list<coords*>* edge_pixel_coords_list = new forward_list<coords*>;
    int imax = len(cell_numbers_data,1)
    int jmax = len(cell_numbers_data,2)
    for (int j = section_min_lon; i <= section_max_lon; i++) {
        for (int i = section_min_lat; i <+ section_max_lat; i++) {
            if (cell_numbers_data(i,j) == cell_number) {
                bool is_edge = false;
                if (i != 1 && j != 1) {
                    is_edge = (cell_numbers_data(i-1,j-1) != cell_number);
                }
                if (i != 1) {
                    is_edge = (cell_numbers_data(i-1,j)   != cell_number) || is_edge;
                }
                if (i != 1 && j != jmax) {
                    is_edge = (cell_numbers_data(i-1,j+1) != cell_number) || is_edge;
                }
                if (j != 1 ) {
                    is_edge = (cell_numbers_data(i,j-1)   != cell_number) || is_edge;
                }
                if (j != jmax) {
                    is_edge = (cell_numbers_data(i,j+1)   != cell_number) || is_edge;
                }
                if (i != imax && j != 1) {
                    is_edge = (cell_numbers_data(i+1,j-1) != cell_number) || is_edge;
                }
                if (i != imax) {
                    is_edge = (cell_numbers_data(i+1,j)   != cell_number) || is_edge;
                }
                if (i != imax && j != jmax) {
                    is_edge = (cell_numbers_data(i+1,j+1) != cell_number) || is_edge;
                }
                if (is_edge) {
                    edge_pixel_coords_list->push_front(new latlon_coords(i,j));
                }
            }
        }
    }
    return edge_pixel_coords_list;
}

double latlon_calculate_length_through_pixel(coords* coords_in) {
    if (is_diagonal(coords_in)) {
        return 1.414;
    } else {
        return 1.0;
    }
}

bool latlon_check_if_coords_are_in_area(coords* coords_in) {
    if (coords_in->get_lat() >= section_min_lat &&
        coords_in->get_lat() <= section_max_lat &&
        coords_in->get_lon() >= section_min_lon &&
        coords_in->get_lon() <= section_max_lon) {
        return true;
    } else {
        return false;
    }
}

bool generic_check_if_coords_are_in_area(coords* coords_in) {
    class(coords), intent(in) :: coords_in
    class(field_section), pointer :: cell_numbers
    type(doubly_linked_list), pointer :: list_of_cells_in_neighborhood
    int working_cell_number;
    bool within_area
    int cell_number
    if ?? class is (irregular_latlon_cell)
        cell_numbers = cell_numbers
        cell_number = cell_number
    class is (irregular_latlon_neighborhood)
        list_of_cells_in_neighborhood = list_of_cells_in_neighborhood
        cell_numbers = cell_numbers
    }
    within_area = false
    if ?? class is (field)
        within_area = true
    class is (cell)
        working_cell_number_ptr = (*cell_numbers)(coords_in);
                working_cell_number = working_cell_number_ptr
        delete working_cell_number_ptr;
        if(working_cell_number == cell_number) {
            within_area = true
        }
    else if class is (neighborhood){
        list_of_cells_in_neighborhood->reset_iterator()
        do while (! list_of_cells_in_neighborhood->iterate_forward()) {
            cell_number_ptr =
                list_of_cells_in_neighborhood->get_value_at_iterator_position()
                working_cell_number_ptr = (*cell_numbers)(coords_in)
                    working_cell_number = working_cell_number_ptr
                delete working_cell_number_ptr;
                if(working_cell_number == cell_number_ptr) {
                    within_area = true
                }
        }
    }
    return within_area;
}

forward_list<coords*>* latlon_find_upstream_neighbors(coords* coords_in) {
    forward_list<coords*>* list_of_neighbors = new forward_list<coords*>;
    int counter = 0;
    for (int i = coords_in->get_lat() + 1, i >= coords_in->get_lat() - 1; i--) {
        for (int j = coords_in->get_lon() - 1; j <= coords_in->get_lon() + 1; i++) {
            counter = counter + 1;
            if (counter == 5) continue;
            if (neighbor_flows_to_pixel(latlon_coords(i,j),
                dir_based_direction_indicator(counter))) {
                list_of_neighbors.push_front(new latlon_coords(lat=i,lon=j))
            }
        }
    }
    return list_of_neighbors;
}

void latlon_initialize_cell_cumulative_flow_subfield() {
    class(*), dimension(:,:), pointer :: input_data
    bool wrap;
    allocate(integer::input_data(section_width_lat,section_width_lon))
        input_data = 0
    (river_directions = river_directions)
    class is (latlon_field_section)
        wrap = (section_width_lon == river_directions->get_nlon() &&
                river_directions->get_wrap())
    (cell_section_coords = cell_section_coords)
    class is (latlon_section_coords)
        cell_cumulative_flow = latlon_subfield(input_data,cell_section_coords,
                                                     wrap)
}

void latlon_initialize_rejected_pixels_subfield() {
    class(*), dimension(:,:), pointer :: rejected_pixel_initialization_data
    bool wrap
    allocate(logical::rejected_pixel_initialization_data(section_width_lat,
                                                         section_width_lon))
        rejected_pixel_initialization_data = false
    (river_directions = river_directions)
    class is (latlon_field_section)
        wrap = (section_width_lon == river_directions->get_nlon() &&
                river_directions->get_wrap())
    (cell_section_coords = cell_section_coords)
    class is (latlon_section_coords)
        rejected_pixels = latlon_subfield(rejected_pixel_initialization_data,cell_section_coords,wrap)
}

forward_list<coords*>* latlon_generate_list_of_pixels_in_cell() {
    vector<coords*>* list_of_pixels_in_cell = new vector<coords*>;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            list_of_pixels_in_cell->push(latlon_coords(i,j));
        }
    }
    return list_of_pixels_in_cell
}

forward_list<coords*>* irregular_latlon_generate_list_of_pixels_in_cell() {
    forward_list<coords*>*: list_of_pixels_in_cell = new forward_list<coords*>;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            if (cell_numbers(i,j) == cell_number) {
                list_of_pixels_in_cell->push_front(new latlon_coords(i,j));
            }
        }
    }
}

int latlon_get_sink_combined_cumulative_flow(coords* main_sink_coords = nullptr) {
    int highest_outflow_sink_lat = 0;
    int highest_outflow_sink_lon = 0;
    int combined_cumulative_flow_to_sinks = 0;
    int greatest_flow_to_single_sink = 0;
    direction_indicator* flow_direction_for_sink = get_flow_direction_for_sink();
    for (int j = section_min_lon; j <= section_min_lon + section_width_lon - 1; i++) {
        for (int i = section_min_lat; i <= section_min_lat + section_width_lat - 1; i++) {
            int current_pixel_total_cumulative_flow =
                (*total_cumulative_flow)(latlon_coords(i,j));
            int current_pixel_flow_direction =
                (*river_directions)(latlon_coords(i,j));
            if (flow_direction_for_sink->is_equal_to(current_pixel_flow_direction)) {
                combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow +
                    combined_cumulative_flow_to_sinks
                if (greatest_flow_to_single_sink <
                    current_pixel_total_cumulative_flow) {
                    current_pixel_total_cumulative_flow =
                        greatest_flow_to_single_sink;
                    highest_outflow_sink_lat = i;
                    highest_outflow_sink_lon = j;
                }
            }
        }
    }
    delete flow_direction_for_sink;
    main_sink_coords = new latlon_coords(highest_outflow_sink_lat,
                                         highest_outflow_sink_lon);
    return combined_cumulative_flow_to_sinks;
}

int latlon_get_rmouth_outflow_combined_cumulative_flow(coords*
                                                       main_outflow_coords =
                                                       nullptr) {
    int highest_outflow_rmouth_lat = 0;
    int highest_outflow_rmouth_lon = 0;
    int combined_cumulative_rmouth_outflow = 0;
    int greatest_flow_to_single_outflow = 0;
    direction_indicator* flow_direction_for_outflow = get_flow_direction_for_outflow();
    for (int j = section_min_lon; j <= section_min_lon + section_width_lon - 1; i++) {
        for (int i = section_min_lat; i <= section_min_lat + section_width_lat - 1; i++) {
            int current_pixel_total_cumulative_flow =
                (*total_cumulative_flow)(latlon_coords(i,j))
            int current_pixel_flow_direction = (*river_directions)(latlon_coords(i,j));
            if (flow_direction_for_outflow->is_equal_to(current_pixel_flow_direction)) {
                combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow +
                    combined_cumulative_rmouth_outflow;
                if (greatest_flow_to_single_outflow <
                    current_pixel_total_cumulative_flow) {
                    current_pixel_total_cumulative_flow =
                        greatest_flow_to_single_outflow;
                    highest_outflow_rmouth_lat = i;
                    highest_outflow_rmouth_lon = j;
                }
            }
        }
    }
    delete flow_direction_for_outflow;
    main_outflow_coords = new latlon_coords(highest_outflow_rmouth_lat,
                                            highest_outflow_rmouth_lon);
    return combined_cumulative_rmouth_outflow;
}

int irregular_latlon_get_sink_combined_cumulative_flow(coords*
                                                       main_sink_coords = nullptr) {
    int highest_outflow_sink_lat = 0;
    int highest_outflow_sink_lon = 0;
    int combined_cumulative_flow_to_sinks = 0;
    int greatest_flow_to_single_sink = 0;
    direction_indicator* flow_direction_for_sink = get_flow_direction_for_sink();
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            if (cell_numbers(new coords(i,j)) == cell_number) {
                int current_pixel_total_cumulative_flow =
                    (*total_cumulative_flow)(latlon_coords(i,j))
                int current_pixel_flow_direction =
                    (*river_directions)(latlon_coords(i,j))
                if (flow_direction_for_sink->is_equal_to(current_pixel_flow_direction)) {
                    combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow +
                        combined_cumulative_flow_to_sinks
                    if (greatest_flow_to_single_sink <
                        current_pixel_total_cumulative_flow) {
                        current_pixel_total_cumulative_flow =
                            greatest_flow_to_single_sink;
                        highest_outflow_sink_lat = i;
                        highest_outflow_sink_lon = j;
                    }
                }
            }
        }
    }
    delete flow_direction_for_sink;
    main_sink_coords = new latlon_coords(highest_outflow_sink_lat,
                                         highest_outflow_sink_lon);
    return combined_cumulative_flow_to_sinks;
}

int irregular_latlon_get_rmouth_outflow_combined_cumulative_flow(coords*
                                                                 main_outflow_coords =
                                                                 nullptr){
    int highest_outflow_rmouth_lat = 0;
    int highest_outflow_rmouth_lon = 0;
    int combined_cumulative_rmouth_outflow = 0;
    int greatest_flow_to_single_outflow = 0;
    direction_indicator* flow_direction_for_outflow =
        get_flow_direction_for_outflow();
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            if (cell_numbers(new latlon_coords(i,j)) == cell_number) {
                int current_pixel_total_cumulative_flow =
                    (*total_cumulative_flow)(latlon_coords(i,j));
                int current_pixel_flow_direction =
                    (*river_directions)(latlon_coords(i,j));
                if (flow_direction_for_outflow->is_equal_to(current_pixel_flow_direction)) {
                    combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow +
                        combined_cumulative_rmouth_outflow;
                    if (greatest_flow_to_single_outflow <
                        current_pixel_total_cumulative_flow) {
                        current_pixel_total_cumulative_flow =
                            greatest_flow_to_single_outflow;
                        highest_outflow_rmouth_lat = i;
                        highest_outflow_rmouth_lon = j;
                    }
                }
            }
        }
    }
    delete flow_direction_for_outflow;
    main_outflow_coords = new latlon_coords(highest_outflow_rmouth_lat,
                                            highest_outflow_rmouth_lon);
    return combined_cumulative_rmouth_outflow;
}


void latlon_print_area(){
    character(len=160) :: line
    cout << "Area Information" << endl;
    cout << "min lat=" << section_min_lat << endl;
    cout << "min lon=" << section_min_lon << endl;
    cout << "max lat=" << section_max_lat << endl;
    cout << "max lon=" << section_max_lon << endl;
    cout << "area rdirs"
    for (int i = section_min_lat; i <= section_max_lat; i++) {
        for( int j = section_min_lon, j <= section_max_lon; j++) {
            cout << (*river_directions)(latlon_coords(i,j))
                cout << value
        }
        cout << endl;
    }
    cout <<  "area total cumulative flow" << endl;
    for (int i = section_min_lat; i <= section_max_lat; i++) {
        for (int j = section_min_lon; j <= section_max_lon; j++) {
            cout << (*total_cumulative_flow)(latlon_coords(i,j))
        }
        cout << endl;
    }
}

double icon_single_index_field_dummy_real_function(coords* coords_in) {
    return 0.0;
}

void icon_single_index_field_dummy_subroutine() {
    throw runtime_error()
}

vector<coords*>* icon_single_index_field_dummy_coords_pointer_function(coords* coords_in) {
    return new vector<coords*>;
}

void init_dir_based_rdirs_latlon_cell(cell_section_coords,river_directions,
                                            total_cumulative_flow) {
    type(latlon_section_coords) :: cell_section_coords
    integer, dimension(:,:) :: river_directions
    integer, dimension(:,:) :: total_cumulative_flow
    init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
    allocate(cell_neighborhood, source=latlon_dir_based_rdirs_neighborhood(cell_section_coords,
             river_directions, total_cumulative_flow))
}

void init_irregular_latlon_dir_based_rdirs_cell(cell_section_coords,river_directions,
                                                      total_cumulative_flow,cell_neighbors){
    type(irregular_latlon_section_coords), intent(in) :: cell_section_coords
    integer, dimension(:,:), intent(in) :: river_directions
    integer, dimension(:,:), intent(in) :: total_cumulative_flow
    integer, dimension(:,:), pointer, intent(in) :: cell_neighbors
    init_irregular_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow)
    allocate(cell_neighborhood,
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
    yamazaki_init_latlon_cell(cell_section_coords,river_directions,total_cumulative_flow,
                                        yamazaki_outlet_pixels)
    allocate(cell_neighborhood, source=
             latlon_dir_based_rdirs_neighborhood(cell_section_coords,river_directions,
             total_cumulative_flow,yamazaki_outlet_pixels,yamazaki_section_coords))
}

void yamazaki_latlon_calculate_river_directions_as_indices(int* coarse_river_direction_indices,
                                                           section_coords*
                                                           required_coarse_section_coords = nullptr) {
    int lat_offset;
    int lon_offset;
    if (required_coarse_section_coords) {
        lat_offset = required_coarse_section_coords->section_min_lat - 1;
        lon_offset = required_coarse_section_coords->section_min_lon - 1;
    } else {
        lat_offset = 0;
        lon_offset = 0;
    }
    int outlet_pixel_type;
    coords* initial_outlet_pixel = yamazaki_retrieve_initial_outlet_pixel(outlet_pixel_type)
    coords* initial_cell_coords = cell_neighborhood->yamazaki_get_cell_coords(initial_outlet_pixel)
    if (outlet_pixel_type == yamazaki_sink_outlet || outlet_pixel_type == yamazaki_rmouth_outlet) {
            coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),1) =
                -abs(outlet_pixel_type);
            coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),2) =
                -abs(outlet_pixel_type);
        delete initial_cell_coords;
        delete initial_outlet_pixel;
        return
    }
    coords* destination_cell_coords = cell_neighborhood->yamazaki_find_downstream_cell(initial_outlet_pixel);
    coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),1) =
        destination_cell_coords->get_lat() - lat_offset;
    coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),2) =
        destination_cell_coords->get_lon() - lon_offset;
    delete initial_cell_coords;
    delete destination_cell_coords;
    delete initial_outlet_pixel;
}

bool latlon_dir_based_rdirs_neighbor_flows_to_pixel(coords* coords_in,
                                                    direction_indicator*
                                                    flipped_direction_from_neighbor) {
    int directions_pointing_to_pixel[9] = { 9,8,7,6,5,4,3,2,1 }
    int rdir = (*river_directions)(coords_in);
        if ( rdir ==
            directions_pointing_to_pixel(flipped_direction_from_neighbor->get_direction())) {
            return true;
        } else {
            return false;
        }
    return neighbor_flows_to_pixel;
}

bool icon_si_index_based_rdirs_neighbor_flows_to_pixel_dummy(coords_in,
    flipped_direction_from_neighbor) {
    return true;
}


direction_indicator*
latlon_dir_based_rdirs_get_flow_direction_for_sink(bool is_for_cell=false) {
    return new dir_based_direction_indicator(flow_direction_for_sink)
}

direction_indicator*
latlon_dir_based_rdirs_get_flow_direction_for_outflow(bool is_for_cell=false) {
    return new dir_based_direction_indicator(flow_direction_for_outflow);
}

direction_indicator*
latlon_dir_based_rdirs_get_flow_direction_for_ocean_point(bool is_for_cell=false) {
    return new dir_based_direction_indicator(flow_direction_for_ocean_point);
}

direction_indicator*
irregular_latlon_dir_based_rdirs_get_flow_direction_for_sink(bool is_for_cell=false) {
    if (is_for_cell) {
        return new index_based_direction_indicator(cell_flow_direction_for_sink);
    } else {
        return new index_based_direction_indicator(flow_direction_for_sink);
    }
}

direction_indicator*
irregular_latlon_dir_based_rdirs_get_flow_direction_for_outflow(bool is_for_cell=false) {
    if (is_for_cell) {
        return new index_based_direction_indicator(cell_flow_direction_for_outflow);
    } else {
        return new index_based_direction_indicator(flow_direction_for_outflow);
    }
}

direction_indicator*
irregular_latlon_dir_based_rdirs_get_flow_direction_for_o_point(bool is_for_cell=false) {
    if (is_for_cell) {
        return new index_based_direction_indicator(cell_flow_direction_for_ocean_point);
    } else {
        return new index_based_direction_indicator(flow_direction_for_ocean_point);
    }
}

bool dir_based_rdirs_is_diagonal(coords* coords_in) {
    int rdir = (*river_directions)(coords_in);
    if (rdir == 7 || rdir == 9 || rdir == 1 || rdir == 3) {
        return true;
    } else {
        return false;
    }
}

coords* icon_si_index_based_rdirs_is_diagonal_dummy(coords* coords_in) {
    return false;
}

void latlon_dir_based_rdirs_mark_ocean_and_river_mouth_points(this) {
    bool non_ocean_cell_found = false;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            int rdir = (*river_directions)(latlon_coords(i,j))
                if( rdir == flow_direction_for_outflow ) {
                    contains_river_mouths = true;
                    ocean_cell = false;
                } else if( rdir != flow_direction_for_ocean_point ) {
                    non_ocean_cell_found = true;
                }
            if (contains_river_mouths) return;
        }
    }
    contains_river_mouths = false;
    ocean_cell = ! non_ocean_cell_found;
}

void irregular_latlon_dir_based_rdirs_mark_o_and_rm_points() {
    bool non_ocean_cell_found = false;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            if (cell_numbers(new latlon_coords(i,j)) == cell_number) {
                int rdir = (*river_directions)(latlon_coords(i,j))
                if (rdir == flow_direction_for_outflow) {
                    contains_river_mouths = true;
                    ocean_cell = false;
                } else if (rdir != flow_direction_for_ocean_point) {
                    non_ocean_cell_found = true;
                }
                if( contains_river_mouths ) return;
            }
        }
    }
    contains_river_mouths = false;
    ocean_cell = ! non_ocean_cell_found;
}

void neighborhood_destructor(this) {
    if (river_directions) delete river_directions;
    if (total_cumulative_flow) delete total_cumulative_flow;
    if (yamazaki_outlet_pixels) delete yamazaki_outlet_pixels;
}

coords* find_downstream_cell(coords* initial_outlet_pixel) {
    int initial_cumulative_flow = (*total_cumulative_flow)(initial_outlet_pixel);
    coords* working_pixel = initial_outlet_pixel->clone();
    bool coords_not_in_neighborhood = false;
    bool outflow_pixel_reached;
    find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
    coords* downstream_cell;
    while(true) {
        downstream_cell = in_which_cell(working_pixel);
        int last_cell_cumulative_flow;
        while(true) {
            last_cell_cumulative_flow = (*total_cumulative_flow)(working_pixel);
            find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,
                                       outflow_pixel_reached)
            if (coords_not_in_neighborhood ||
                downstream_cell != in_which_cell(working_pixel) ||
                outflow_pixel_reached) break;
        }
        if (coords_not_in_neighborhood || outflow_pixel_reached) break;
        if ((last_cell_cumulative_flow  - initial_cumulative_flow > area_threshold) &&
            ( ! is_center_cell(downstream_cell))) break;
    }
    delete last_cell_cumulative_flow;
    delete working_pixel;
    return downstream_cell
}

coords* yamazaki_find_downstream_cell(coords* initial_outlet_pixel) {
    bool downstream_cell_found = false;
    coords* working_pixel = initial_outlet_pixel->clone();
    coords* cell_coords = yamazaki_get_cell_coords(initial_outlet_pixel);
    double downstream_path_length = 0.0;
    int count_of_cells_passed_through = 0;
    while (true) {
        bool outflow_pixel_reached;
        bool coords_not_in_neighborhood;
        find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
        downstream_path_length = downstream_path_length + calculate_length_through_pixel(working_pixel)
        if (coords_not_in_neighborhood) yamazaki_wrap_coordinates(working_pixel)
        coords* new_cell_coords = yamazaki_get_cell_coords(working_pixel)
        if ((*cell_coords) != (*new_cell_coords)) {
            delete cell_coords;
            cell_coords = new_cell_coords->clone();
            count_of_cells_passed_through = count_of_cells_passed_through + 1
        }
        delete new_cell_coords;
        ??? yamazaki_outlet_pixel_type = (*yamazaki_outlet_pixels)(working_pixel);
            if (yamazaki_outlet_pixel_type == yamazaki_normal_outlet &&
                downstream_path_length > MUFP) downstream_cell_found = true
            if (yamazaki_outlet_pixel_type == yamazaki_rmouth_outlet ||
                yamazaki_outlet_pixel_type == yamazaki_sink_outlet)
                downstream_cell_found = true
        if (downstream_cell_found) break;
        if (count_of_cells_passed_through > yamazaki_max_range) {
            delete working_pixel;
            working_pixel = initial_outlet_pixel->clone();
            find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,outflow_pixel_reached)
            if (coords_not_in_neighborhood) yamazaki_wrap_coordinates(working_pixel)
            delete cell_coords;
            cell_coords = yamazaki_get_cell_coords(working_pixel)
            break;
        }
    }
    delete working_pixel;
    return cell_coords;
}

bool path_reenters_region(coords* coords_inout) {
    while(true) {
        bool coords_not_in_neighborhood;
        find_next_pixel_upstream(coords_inout,coords_not_in_neighborhood)
        if (coords_not_in_neighborhood) {
            return false;
        } else if (in_which_cell(coords_inout) == center_cell_id) {
            return true;
        }
    }
}

bool is_center_cell(int cell_id) {
    if (cell_id == center_cell_id) return true;
    else return false;
}

void init_latlon_neighborhood(center_cell_section_coords,river_directions,
                                    total_cumulative_flow) {
    type(latlon_section_coords) :: center_cell_section_coords
    type(latlon_section_coords) :: section_coords
    integer, target, dimension(:,:) :: river_directions
    integer, target, dimension(:,:) :: total_cumulative_flow
    center_cell_min_lat = center_cell_section_coords->section_min_lat;
    center_cell_min_lon = center_cell_section_coords->section_min_lon;
    center_cell_max_lat =  center_cell_section_coords->section_min_lat +
                                center_cell_section_coords->section_width_lat - 1;
    center_cell_max_lon =  center_cell_section_coords->section_min_lon +
                                center_cell_section_coords->section_width_lon - 1;
    section_min_lat = center_cell_section_coords->section_min_lat -
                           center_cell_section_coords->section_width_lat;
    section_min_lon = center_cell_section_coords->section_min_lon -
                           center_cell_section_coords->section_width_lon;
    section_max_lat = center_cell_max_lat +
                           center_cell_section_coords->section_width_lat;
    section_max_lon = center_cell_max_lon +
                           center_cell_section_coords->section_width_lon;
    section_coords = latlon_section_coords(section_min_lat,section_min_lon,
                                           section_max_lat,section_max_lon)
    river_directions = latlon_field_section(river_directions,
                                                  section_coords)
    total_cumulative_flow = latlon_field_section(total_cumulative_flow,
                                                       section_coords)
    center_cell_id = 5;
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
    class(*), dimension(:,:), pointer :: cell_numbers_pointer
    class(*), pointer :: cell_number_pointer
    int nbr_cell_number
    int max_nneigh_coarse = len(coarse_cell_neighbors,2)
    center_cell_id = center_cell_section_coords->cell_number
    allocate(list_of_cells_in_neighborhood)
    int nneigh_coarse = 0;
    for (int i=1; i <= max_nneigh_coarse; i++ {
        nbr_cell_number = coarse_cell_neighbors(center_cell_id,i)
        if (nbr_cell_number > 0) {
            list_of_cells_in_neighborhood->
                push(nbr_cell_number)
            nneigh_coarse = nneigh_coarse+1
        }
    }
    allocate(list_of_cell_numbers(nneigh_coarse+1))
    list_of_cell_numbers(1) = center_cell_id
    list_of_cells_in_neighborhood->reset_iterator()
    int i = 1
    do while (! list_of_cells_in_neighborhood->iterate_forward()) {
        i = i + 1
        cell_number_pointer = list_of_cells_in_neighborhood->get_value_at_iterator_position()
            list_of_cell_numbers(i) = cell_number_pointer
    }
    neighborhood_section_coords =
        irregular_latlon_section_coords(list_of_cell_numbers(i),
                                        center_cell_section_coords->cell_numbers,
                                        center_cell_section_coords->section_min_lats,
                                        center_cell_section_coords->section_min_lons,
                                        center_cell_section_coords->section_max_lats,
                                        center_cell_section_coords->section_max_lons )
    cell_numbers_pointer = center_cell_section_coords->cell_numbers;
    cell_numbers = latlon_field_section(cell_numbers_pointer,neighborhood_section_coords);
    river_directions = latlon_field_section(river_directions,
                                            neighborhood_section_coords);
    total_cumulative_flow = latlon_field_section(total_cumulative_flow,
                                                 neighborhood_section_coords);
    neighborhood_section_coords->irregular_latlon_section_coords_destructor();
    delete list_of_cell_numbers;
}

void irregular_latlon_neighborhood_destructor() {
    if (list_of_cells_in_neighborhood) {
        list_of_cells_in_neighborhood->reset_iterator()
        do while ( ! list_of_cells_in_neighborhood->iterate_forward()) {
            list_of_cells_in_neighborhood->remove_element_at_iterator_position()
        }
        delete list_of_cells_in_neighborhood;
    }
    if (cell_numbers) delete cell_numbers;
}

void yamazaki_init_latlon_neighborhood(latlon_section_coords* center_cell_section_coords,
                                       int* river_directions,
                                       int* total_cumulative_flow,
                                       int* yamazaki_outlet_pixels_pointer,
                                       latlon_section_coords* yamazaki_section_coords) {
    init_latlon_neighborhood(center_cell_section_coords,river_directions,
                             total_cumulative_flow);
    section_min_lat = yamazaki_section_coords->section_min_lat;
    section_min_lon = yamazaki_section_coords->section_min_lon;
    section_max_lat = yamazaki_section_coords->section_min_lat +
                      yamazaki_section_coords->section_width_lat - 1;
    section_max_lon = yamazaki_section_coords->section_min_lon +
                      yamazaki_section_coords->section_width_lon - 1;
    yamazaki_outlet_pixels = latlon_field_section(yamazaki_outlet_pixels,
                                                  yamazaki_section_coords);
    yamazaki_cell_width_lat = center_cell_max_lat + 1 - center_cell_min_lat;
    yamazaki_cell_width_lon = center_cell_max_lon + 1 - center_cell_min_lon;
}

int latlon_in_which_cell(coords* pixel) {
    int internal_cell_lat;
    int internal_cell_lon;
    if (pixel->get_lat() < center_cell_min_lat) {
        internal_cell_lat = 1;
    } else if (pixel->get_lat() > center_cell_max_lat) {
        internal_cell_lat = 3;
    } else {
        internal_cell_lat = 2;
    }
    if (pixel->get_lon() < center_cell_min_lon) {
        internal_cell_lon = 1;
    } else if (pixel->get_lon() > center_cell_max_lon) {
        internal_cell_lon = 3;
    } else {
        internal_cell_lon = 2;
    }
    return internal_cell_lon + 9 - 3*internal_cell_lat;
}

int irregular_latlon_in_which_cell(coords* pixel) {
    return (*cell_numbers)(pixel);
}

coords* yamazaki_latlon_get_cell_coords(coords* pixel_coords) {
    return new latlon_coords(ceil((double)(pixel_coords->get_lat())/yamazaki_cell_width_lat),
                             ceil((double)(pixel_coords->get_lon())/yamazaki_cell_width_lon));
}


coords* yamazaki_irregular_latlon_get_cell_coords_dummy(coords* pixel_coords) {
    return nullptr;
}

//This will neeed updating if we use neighborhood smaller than the full grid size for the
//yamazaki-style algorithm - at the moment it assumes at neighborhood edges are edges of the
//full latitude-longitude grid and need longitudinal wrapping.
void yamazaki_latlon_wrap_coordinates(coords* pixel_coords){
    if ( ! yamazaki_wrap ) return
    if (pixel_coords->get_lon() < section_min_lon) {
        pixel_coords->lon = section_max_lon + pixel_coords->get_lon() + 1 - section_min_lon;
    } else if (pixel_coords->get_lon() > section_max_lon) {
        pixel_coords->lon = section_min_lon + pixel_coords->get_lon() - 1 - section_max_lon;
    }
}

void yamazaki_irregular_latlon_wrap_coordinates_dummy(coords* pixel_coords){}

void latlon_dir_based_rdirs_find_next_pixel_downstream(coords* coords_inout,
                                                       bool& coords_not_in_area,
                                                       bool& outflow_pixel_reached = false) {
    int rdir = (*river_directions)(coords_inout);
    if ( rdir == 7 || rdir == 8 || rdir == 9) {
        coords_inout->lat = coords_inout->get_lat() - 1;
    } else if ( rdir == 1 || rdir == 2 || rdir == 3) {
        coords_inout->lat = coords_inout->get_lat() + 1;
    }
    if ( rdir == 7 || rdir == 4 || rdir == 1 ) {
        coords_inout->lon = coords_inout->get_lon() - 1;
    } else if ( rdir == 9 || rdir == 6 || rdir == 3 ) {
        coords_inout->lon = coords_inout->get_lon() + 1;
    }
    if ( rdir == 5 || rdir == 0 || rdir == -1 ) {
        coords_not_in_area = false;
        outflow_pixel_reached =  true;
    } else {
        outflow_pixel_reached =  false;
        if (check_if_coords_are_in_area(coords_inout)) {
            coords_not_in_area = false;
        } else {
            coords_not_in_area = true;
        }
    }
}

void icon_single_index_index_based_rdirs_find_next_pixel_downstream(
        coords* coords_inout,
        bool& coords_not_in_area,
        bool& outflow_pixel_reached = false) {
    coords* next_cell_index = (*river_directions)(coords_inout);
    if ( next_cell_index == index_for_sink ||
         next_cell_index == index_for_outflow ||
         next_cell_index == index_for_ocean_point ) {
        coords_not_in_area = false;
        outflow_pixel_reached =  true;
    } else {
        outflow_pixel_reached =  false;
        if (check_if_coords_are_in_area(coords_inout)) {
            coords_inout->index = next_cell_index;
            coords_not_in_area = false;
        } else {
            coords_not_in_area = true;
        }
    }
    delete next_cell_index;
}

direction_indicator* dir_based_rdirs_calculate_direction_indicator(int downstream_cell) {
    return new dir_based_direction_indicator(downstream_cell);
}

direction_indicator* irregular_latlon_calculate_direction_indicator(int downstream_cell) {
    return new index_based_direction_indicator(downstream_cell);
}
