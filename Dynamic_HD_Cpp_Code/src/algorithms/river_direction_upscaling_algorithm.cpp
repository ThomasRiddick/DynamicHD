#include "algorithms/river_direction_upscaling_algorithm.hpp"
#include "base/coords.hpp"
#include "base/field.hpp"
#include "base/subfield.hpp"
#include <stdexcept>
#include <cmath>

area::~area() {
    if (river_directions) delete river_directions;
    if (total_cumulative_flow) delete total_cumulative_flow;
}

cell::~cell() {
    if (rejected_pixels) delete rejected_pixels;
    if (cell_cumulative_flow) delete cell_cumulative_flow;
    if (cell_neighborhood) delete cell_neighborhood;
    if (delete_parameters_on_exit) delete parameters;
}

neighborhood::~neighborhood() {
    if (neighborhood_section_coords) delete neighborhood_section_coords;
    if (delete_parameters_on_exit) delete parameters;
}

extended_area::~extended_area() {
    if (cells_to_reprocess) delete cells_to_reprocess;
}

irregular_latlon_cell::~irregular_latlon_cell() {
    if (cell_numbers) delete cell_numbers;
}

irregular_latlon_neighborhood::~irregular_latlon_neighborhood() {
    if (cell_numbers) delete cell_numbers;
}

void area::find_next_pixel_upstream(coords*& coords_inout,
                                    bool& coords_not_in_area){
    int max_cmltv_flow_working_value = 0;
    list<coords*>* upstream_neighbors_coords_list = find_upstream_neighbors(coords_inout);
    if (! upstream_neighbors_coords_list->empty()) {
        for(list<coords*>::iterator i = upstream_neighbors_coords_list->begin();
                                      i != upstream_neighbors_coords_list->end(); ++i){
            int upstream_neighbor_total_cumulative_flow =
                (*total_cumulative_flow)(*i);
            if ( upstream_neighbor_total_cumulative_flow > max_cmltv_flow_working_value) {
                if (coords_inout) delete coords_inout;
                coords_inout = (*i);
                max_cmltv_flow_working_value =
                    upstream_neighbor_total_cumulative_flow;
            } else delete *i;
        }
        coords_not_in_area = ! check_if_coords_are_in_area(coords_inout);
    } else {
        coords_not_in_area = true;
    }
    delete upstream_neighbors_coords_list;
}

coords* cell::find_outlet_pixel(bool& no_remaining_outlets,
                                bool use_LCDA_criterion,
                                bool& outlet_is_LCDA){
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

void cell::check_for_sinks_and_rmouth_outflows(coords*& outlet_pixel_coords,
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
    if (parameters->run_check_for_sinks) {
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
         if (outlet_pixel_coords) delete outlet_pixel_coords;
         if (contains_river_mouths) delete main_river_mouth_coords;
         outlet_pixel_coords =  main_sink_coords;
    } else if (  cumulative_rmouth_outflow > outlet_pixel_cumulative_flow) {
         mark_outflow = true;
         if (outlet_pixel_coords) delete outlet_pixel_coords;
         if (parameters->run_check_for_sinks) delete main_sink_coords;
         outlet_pixel_coords =  main_river_mouth_coords;
    } else {
        if (contains_river_mouths) delete main_river_mouth_coords;
        if (parameters->run_check_for_sinks) delete main_sink_coords;
    }
}


direction_indicator* cell::find_cell_flow_direction(coords*& outlet_pixel_coords,
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
    int downstream_cell =
        cell_neighborhood->find_downstream_cell(outlet_pixel_coords);
    return cell_neighborhood->calculate_direction_indicator(downstream_cell);
}

direction_indicator* cell::process_cell(){
    direction_indicator* flow_direction;
    if (! ocean_cell) {
        bool use_LCDA_criterion = true;
        bool no_remaining_outlets;
        //Required by function but not used
        bool outlet_is_LCDA;
        coords* outlet_pixel_coords = find_outlet_pixel(no_remaining_outlets,
                                                        use_LCDA_criterion,
                                                        outlet_is_LCDA);
        flow_direction = find_cell_flow_direction(outlet_pixel_coords,
                                                  no_remaining_outlets);
        if (parameters->upscale_k_parameters) calculate_k_parameters(outlet_pixel_coords);
        delete outlet_pixel_coords;
    } else {
        bool is_for_cell = true;
        flow_direction = get_flow_direction_for_ocean_point(is_for_cell);
    }
    return flow_direction;
}

void cell::calculate_k_parameters(coords* outlet_pixel_coords) {
    coords* working_coords = outlet_pixel_coords;
    bool coords_not_in_cell = false;
    while ( ! coords_not_in_cell) {
        find_next_pixel_upstream(working_coords,
                                 coords_not_in_cell);
    }
}

void cell::flow_mark_cell_outlet_pixel(bool use_LCDA_criterion) {
    if (! ocean_cell) {
        bool outlet_is_LCDA = false;
        bool no_remaining_outlets;
        coords* outlet_pixel_coords = find_outlet_pixel(no_remaining_outlets,use_LCDA_criterion,
                                                        outlet_is_LCDA);
        bool mark_sink;
        bool mark_outflow;
        check_for_sinks_and_rmouth_outflows(outlet_pixel_coords,no_remaining_outlets,mark_sink,
                                            mark_outflow);
        pixel_type outlet_type;
        if (mark_sink) {
            outlet_type = flow_sink_outlet;
        } else if (mark_outflow) {
            outlet_type = flow_rmouth_outlet;
        } else {
            if ( ! (no_remaining_outlets || outlet_is_LCDA) ) {
                outlet_type = flow_normal_outlet;
            } else {
                outlet_type = flow_source_cell_outlet;
            }
        }
        (*flow_outlet_pixels)(outlet_pixel_coords) = outlet_type;
        delete outlet_pixel_coords;
    }
}

// void latlon_cell::print_cell_and_neighborhood(){
//     cout << "For cell" << endl;
//     print_area();
//     cout <<  "contains_river_mouths= " << contains_river_mouths << endl;
//     cout <<  "ocean_cell= " << ocean_cell << endl;
//     cout << "For neighborhood" << endl;
//     cell_neighborhood->print_area();
// }

coords* cell::flow_retrieve_initial_outlet_pixel(pixel_type& outlet_pixel_type) {
    forward_list<coords*>* edge_pixel_coords_list = find_edge_pixels();
    bool pixel_found = false;
    coords* initial_outlet_pixel;
    for(forward_list<coords*>::iterator i = edge_pixel_coords_list->begin();
        i != edge_pixel_coords_list->end(); ++i) {
        if ( ! pixel_found ) {
            pixel_type working_pixel_value = (*flow_outlet_pixels)(*i);
            if (working_pixel_value == flow_source_cell_outlet ||
                working_pixel_value == flow_normal_outlet ||
                working_pixel_value == flow_sink_outlet ||
                working_pixel_value == flow_rmouth_outlet) {
                outlet_pixel_type = working_pixel_value;
                initial_outlet_pixel = (*i)->clone();
                pixel_found = true;
            }
        }
    }
    delete edge_pixel_coords_list;
    if ( ! pixel_found ) {
        list<coords*>* pixels_to_process = generate_list_of_pixels_in_cell();
        while(true) {
            //Combine iteration and check to see if we have reached the end
            //of the list
            if (pixels_to_process->empty()) {
                if (pixel_found) {
                    break;
                } else {
                    throw runtime_error("no outlet pixel could "
                                        "be retrieved for cell");
                }
            }
            coords* working_pixel =
                pixels_to_process->front();
            if ( ! pixel_found) {
                pixel_type working_pixel_value = (*flow_outlet_pixels)(working_pixel);
                    if (working_pixel_value == flow_sink_outlet ||
                        working_pixel_value == flow_rmouth_outlet) {
                        outlet_pixel_type = working_pixel_value;
                        initial_outlet_pixel = working_pixel->clone();
                        pixel_found = true;
                    }
            }
            pixels_to_process->pop_front();
        }
    }
    return initial_outlet_pixel;
}

coords* cell::find_pixel_with_LUDA(bool& no_remaining_outlets) {
    int LUDA_working_value = 0;
    coords* LUDA_pixel_coords = nullptr;
    forward_list<coords*>* edge_pixel_coords_list = find_edge_pixels();
    coords* first_entry=edge_pixel_coords_list->front()->clone();
    for(forward_list<coords*>::iterator i = edge_pixel_coords_list->begin();
        i != edge_pixel_coords_list->end(); ++i) {
        bool rejected_pixel = (*rejected_pixels)(*i);
        if ( ! rejected_pixel) {
            int upstream_drainage = (*total_cumulative_flow)(*i);
            if (upstream_drainage > LUDA_working_value) {
                coords* working_pixel_coords=(*i)->clone();
                bool coords_not_in_cell;
                bool outflow_pixel_reached;
                working_pixel_coords =
                    find_next_pixel_downstream(working_pixel_coords,
                                               coords_not_in_cell,
                                               outflow_pixel_reached);
                delete working_pixel_coords;
                if (coords_not_in_cell) {
                    LUDA_working_value = upstream_drainage;
                    if (LUDA_pixel_coords) delete LUDA_pixel_coords;
                    LUDA_pixel_coords=(*i)->clone();
                }
            }
        }
        delete *i;
    }
    if (! LUDA_pixel_coords) {
        LUDA_pixel_coords=first_entry;
        no_remaining_outlets=true;
    } else {
        delete first_entry;
        no_remaining_outlets=false;
    }
    delete edge_pixel_coords_list;
    return LUDA_pixel_coords;
}

coords* cell::find_pixel_with_LCDA() {
    int LCDA_pixel_working_value = 0;
    coords* LCDA_pixel_coords = nullptr;
    forward_list<coords*>* edge_pixel_coords_list = find_edge_pixels();
    coords* first_entry=edge_pixel_coords_list->front()->clone();
    for(forward_list<coords*>::iterator i = edge_pixel_coords_list->begin();
        i != edge_pixel_coords_list->end(); ++i) {
        int cell_drainage_area = (*cell_cumulative_flow)(*i);
        if (cell_drainage_area > LCDA_pixel_working_value) {
            coords* working_pixel_coords=(*i)->clone();
            bool coords_not_in_cell;
            bool outflow_pixel_reached;
            working_pixel_coords = find_next_pixel_downstream(working_pixel_coords,
                                                              coords_not_in_cell,
                                                              outflow_pixel_reached);
            delete working_pixel_coords;
            if(coords_not_in_cell) {
                LCDA_pixel_working_value = cell_drainage_area;
                if (LCDA_pixel_coords) delete LCDA_pixel_coords;
                LCDA_pixel_coords=(*i)->clone();
            }
        }
        delete *i;
    }
    if (! LCDA_pixel_coords) {
        LCDA_pixel_coords= first_entry;
    } else delete first_entry;
    delete edge_pixel_coords_list;
    return LCDA_pixel_coords;
}

double cell::measure_upstream_path(coords* outlet_pixel_coords) {
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

bool cell::check_MUFP(coords* LUDA_pixel_coords,
                      coords* LCDA_pixel_coords,
                      bool& LUDA_is_LCDA) {
    LUDA_is_LCDA = false;
    if (measure_upstream_path(LUDA_pixel_coords) > parameters->MUFP) {
        return true;
    } else if (LCDA_pixel_coords) {
        if((*LUDA_pixel_coords)==(*LCDA_pixel_coords)) {
            LUDA_is_LCDA = true;
            return true;
        }
    }
    (*rejected_pixels)(LUDA_pixel_coords) = true;
    return false;
}

coords* cell::flow_test_find_downstream_cell(coords* initial_outlet_pixel) {
    return cell_neighborhood->flow_find_downstream_cell(initial_outlet_pixel);
}

int* cell::test_generate_cell_cumulative_flow() {
    generate_cell_cumulative_flow();
    return cell_cumulative_flow->get_array();
}

void cell::generate_cell_cumulative_flow() {
    initialize_cell_cumulative_flow_subfield();
    list<coords*>* pixels_to_process = generate_list_of_pixels_in_cell();
    while(! pixels_to_process->empty()) {
        for (list<coords*>::iterator i = pixels_to_process->begin();
             i != pixels_to_process->end();){
            int pixel_cumulative_flow = 1;
            bool unprocessed_inflowing_pixels = false;
            list<coords*>* upstream_neighbors_coords_list =
                find_upstream_neighbors(*i);
            if (! upstream_neighbors_coords_list->empty()) {
                for(list<coords*>::iterator j = upstream_neighbors_coords_list->begin();
                    j != upstream_neighbors_coords_list->end(); ++j) {
                    if (! unprocessed_inflowing_pixels) {
                        if( ! check_if_coords_are_in_area(*j)) {
                            delete *j;
                            continue;
                        }
                        int neighbor_cumulative_flow = (*cell_cumulative_flow)(*j);
                        if (neighbor_cumulative_flow == 0) {
                            unprocessed_inflowing_pixels = true;
                        } else {
                            pixel_cumulative_flow = pixel_cumulative_flow +
                                                    neighbor_cumulative_flow;
                        }
                    }
                    delete *j;
                }
            }
            if (! unprocessed_inflowing_pixels) {
                (*cell_cumulative_flow)(*i) = pixel_cumulative_flow;
                delete *i;
                i = pixels_to_process->erase(i);
            } else i++;
            delete upstream_neighbors_coords_list;
        }
    }
    delete pixels_to_process;
}

void cell::set_contains_river_mouths(bool value) {
    contains_river_mouths = value;
}

latlon_cell::latlon_cell() {
    cell_cumulative_flow = nullptr;
    cell_section_coords = nullptr;
    _grid_params = nullptr;
    parameters = nullptr;
    river_directions = nullptr;
    total_cumulative_flow =  nullptr;
    rejected_pixels = nullptr;
}

latlon_cell::latlon_cell(latlon_section_coords* cell_section_coords_in,
                         grid_params* _grid_params_in,
                         int* river_directions_in,
                         int* total_cumulative_flow_in,
                         rdir_upscaling_parameters* parameters_in) {
    cell_cumulative_flow = nullptr;
    cell_section_coords = cell_section_coords_in;
    _grid_params = _grid_params_in;
    if (parameters_in) {
        parameters = parameters_in;
        delete_parameters_on_exit = false;
    } else {
        parameters = new rdir_upscaling_parameters();
        delete_parameters_on_exit = true;
    }
    section_min_lat = cell_section_coords_in->section_min_lat;
    section_min_lon = cell_section_coords_in->section_min_lon;
    section_width_lat = cell_section_coords_in->section_width_lat;
    section_width_lon = cell_section_coords_in->section_width_lon;
    section_max_lat =  cell_section_coords_in->section_min_lat +
                       cell_section_coords_in->section_width_lat - 1;
    section_max_lon =  cell_section_coords_in->section_min_lon +
                       cell_section_coords_in->section_width_lon - 1;
    river_directions = new latlon_field_section<int>(river_directions_in,
                                                cell_section_coords_in,
                                                _grid_params);
    total_cumulative_flow = new latlon_field_section<int>(total_cumulative_flow_in,
                                                          cell_section_coords_in,
                                                          _grid_params);
    //Minus four to avoid double counting the corners
    number_of_edge_pixels = 2*section_width_lat + 2*section_width_lon - 4;
    initialize_rejected_pixels_subfield();
    contains_river_mouths = false;
    ocean_cell = false;
}

irregular_latlon_cell::irregular_latlon_cell(
                            irregular_latlon_section_coords* cell_section_coords_in,
                            grid_params* _grid_params_in,
                            int* river_directions_in,
                            int* total_cumulative_flow_in,
                            rdir_upscaling_parameters* parameters_in) :
        latlon_cell() {
    cell_cumulative_flow = nullptr;
    _grid_params = _grid_params_in;
    if (parameters_in) {
        parameters = parameters_in;
        delete_parameters_on_exit = false;
    } else {
        parameters = new rdir_upscaling_parameters();
        delete_parameters_on_exit = true;
    }
    cell_section_coords = cell_section_coords_in;
    section_min_lat = cell_section_coords_in->section_min_lat;
    section_min_lon = cell_section_coords_in->section_min_lon;
    section_width_lat = cell_section_coords_in->section_width_lat;
    section_width_lon = cell_section_coords_in->section_width_lon;
    section_max_lat =  cell_section_coords_in->section_min_lat +
                       cell_section_coords_in->section_width_lat - 1;
    section_max_lon =  cell_section_coords_in->section_min_lon +
                       cell_section_coords_in->section_width_lon - 1;
    cell_numbers = new latlon_field_section<int>(cell_section_coords_in->cell_numbers,
                                                 cell_section_coords_in,
                                                 _grid_params);
    cell_number = cell_section_coords_in->cell_number;
    river_directions = new latlon_field_section<int>(river_directions_in,
                                                     cell_section_coords_in,
                                                     _grid_params);
    total_cumulative_flow = new latlon_field_section<int>(total_cumulative_flow_in,
                                                          cell_section_coords_in,
                                                          _grid_params);
    number_of_edge_pixels = 0;
    initialize_rejected_pixels_subfield();
    contains_river_mouths = false;
    ocean_cell = false;
}

// void latlon_cell::flow_init_latlon_cell(coords* cell_section_coords,
//                                         int* river_directions,
//                                         int* total_cumulative_flow,
//                                         int* flow_outlet_pixels) {
//     type(latlon_section_coords) :: cell_section_coords
//     integer, dimension(:,:) :: river_directions
//     integer, dimension(:,:) :: total_cumulative_flow
//     integer, dimension(:,:), target :: flow_outlet_pixels
//     init_latlon_cell(cell_section_coords,river_directions,
//                                total_cumulative_flow)
//     flow_outlet_pixels = latlon_field_section(flow_outlet_pixels,
//                                                         cell_section_coords)
// }

forward_list<coords*>* latlon_cell::find_edge_pixels(){
    forward_list<coords*>* edge_pixel_coords_list = new forward_list<coords*>;
    for (int i = section_min_lat; i <= section_max_lat; i++) {
        edge_pixel_coords_list->push_front(new latlon_coords(i,section_min_lon));
        edge_pixel_coords_list->push_front(new latlon_coords(i,section_max_lon));
    }
    for (int i = section_min_lon+1; i <= section_max_lon-1; i++) {
        edge_pixel_coords_list->push_front(new latlon_coords(section_min_lat,i));
        edge_pixel_coords_list->push_front(new latlon_coords(section_max_lat,i));
    }
    return edge_pixel_coords_list;
}

forward_list<coords*>* irregular_latlon_cell::find_edge_pixels() {
    forward_list<coords*>* edge_pixel_coords_list = new forward_list<coords*>;
    latlon_grid_params* latlon_fine_grid_params =
        static_cast<latlon_grid_params*>(_grid_params);
    int imax = latlon_fine_grid_params->get_nlat() - 1;
    int jmax = latlon_fine_grid_params->get_nlon() - 1;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            latlon_coords* neighbor_coords = new latlon_coords();
            if ((*cell_numbers)(working_coords) == cell_number) {
                bool is_edge = false;
                if (i != 0 && j != 0) {
                    neighbor_coords->set_coords(i-1,j-1);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number);
                }
                if (i != 0) {
                    neighbor_coords->set_coords(i-1,j);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (i != 0 && j != jmax) {
                    neighbor_coords->set_coords(i-1,j+1);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (j != 0 ) {
                    neighbor_coords->set_coords(i,j-1);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (j != jmax) {
                    neighbor_coords->set_coords(i,j+1);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (i != imax && j != 0) {
                    neighbor_coords->set_coords(i+1,j-1);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (i != imax) {
                    neighbor_coords->set_coords(i+1,j);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (i != imax && j != jmax) {
                    neighbor_coords->set_coords(i+1,j+1);
                    is_edge = ((*cell_numbers)(neighbor_coords) != cell_number) || is_edge;
                }
                if (is_edge) {
                    edge_pixel_coords_list->push_front(working_coords);
                } else delete working_coords;
            } else delete working_coords;
            delete neighbor_coords;
        }
    }
    return edge_pixel_coords_list;
}

double latlon_area::calculate_length_through_pixel(coords* coords_in) {
    if (is_diagonal(coords_in)) {
        return 1.414;
    } else {
        return 1.0;
    }
}

bool latlon_area::check_if_coords_are_in_area(coords* coords_in) {
    latlon_coords* working_coords = static_cast<latlon_coords*>(coords_in);
    if (working_coords->get_lat() >= section_min_lat &&
        working_coords->get_lat() <= section_max_lat &&
        working_coords->get_lon() >= section_min_lon &&
        working_coords->get_lon() <= section_max_lon) {
        return true;
    } else {
        return false;
    }
}



list<coords*>* latlon_area::find_upstream_neighbors(coords* coords_in) {
    latlon_coords* working_coords = static_cast<latlon_coords*>(coords_in);
    list<coords*>* list_of_neighbors = new list<coords*>;
    latlon_grid_params* latlon_fine_grid_params =
        static_cast<latlon_grid_params*>(_grid_params);
    int counter = 0;
    for (int i = working_coords->get_lat() + 1; i >= working_coords->get_lat() - 1; i--) {
        for (int j = working_coords->get_lon() - 1; j <= working_coords->get_lon() + 1; j++) {
            if (i < 0 ||
                i >= latlon_fine_grid_params->get_nlat()) continue;
            int new_j;
            if (! _grid_params->get_nowrap()){
                if (j < 0) {
                    new_j = j + latlon_fine_grid_params->get_nlon();
                } else if (j >= latlon_fine_grid_params->get_nlon()) {
                    new_j = j - latlon_fine_grid_params->get_nlon();
                } else {
                    new_j = j;
                }
            } else {
                new_j = j;
            }
            coords* working_coords = new latlon_coords(i,new_j);
            counter = counter + 1;
            direction_indicator* direction =
                new dir_based_direction_indicator(counter);
            if (counter == 5) {
                delete working_coords;
                delete direction;
                continue;
            }
            if (neighbor_flows_to_pixel(working_coords,direction)) {
                list_of_neighbors->push_front(working_coords);
            } else delete working_coords;
            delete direction;
        }
    }
    return list_of_neighbors;
}

void latlon_cell::initialize_cell_cumulative_flow_subfield() {
    bool wrap = (section_width_lon ==
                 static_cast<latlon_grid_params*>(_grid_params)->get_nlon() &&
                 ! static_cast<latlon_grid_params*>(_grid_params)->get_nowrap());
    cell_cumulative_flow = new latlon_subfield<int>(cell_section_coords,
                                                    wrap);
    cell_cumulative_flow->set_all(0);
}

void latlon_cell::initialize_rejected_pixels_subfield() {
    bool wrap = (section_width_lon ==
                 static_cast<latlon_grid_params*>(_grid_params)->get_nlon() &&
                 ! static_cast<latlon_grid_params*>(_grid_params)->get_nowrap());
    rejected_pixels = new latlon_subfield<bool>(cell_section_coords,wrap);
    rejected_pixels->set_all(false);
}

list<coords*>* latlon_cell::generate_list_of_pixels_in_cell() {
    list<coords*>* list_of_pixels_in_cell = new list<coords*>;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            list_of_pixels_in_cell->push_back(new latlon_coords(i,j));
        }
    }
    return list_of_pixels_in_cell;
}

list<coords*>* irregular_latlon_cell::generate_list_of_pixels_in_cell() {
    list<coords*>* list_of_pixels_in_cell = new list<coords*>;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            if ((*cell_numbers)(working_coords) == cell_number) {
                list_of_pixels_in_cell->push_front(working_coords);
            } else delete working_coords;
        }
    }
    return list_of_pixels_in_cell;
}

int latlon_cell::get_sink_combined_cumulative_flow(coords*& main_sink_coords) {
    int highest_outflow_sink_lat = 0;
    int highest_outflow_sink_lon = 0;
    int combined_cumulative_flow_to_sinks = 0;
    int greatest_flow_to_single_sink = 0;
    direction_indicator* flow_direction_for_sink = get_flow_direction_for_sink();
    for (int j = section_min_lon; j <= section_min_lon + section_width_lon - 1; j++) {
        for (int i = section_min_lat; i <= section_min_lat + section_width_lat - 1; i++) {
            coords* working_coords = new latlon_coords(i,j);
            int current_pixel_total_cumulative_flow =
                (*total_cumulative_flow)(working_coords);
            int current_pixel_flow_direction =
                (*river_directions)(working_coords);
            if (flow_direction_for_sink->is_equal_to(current_pixel_flow_direction)) {
                combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow +
                    combined_cumulative_flow_to_sinks;
                if (greatest_flow_to_single_sink <
                    current_pixel_total_cumulative_flow) {
                    greatest_flow_to_single_sink =
                        current_pixel_total_cumulative_flow;
                    highest_outflow_sink_lat = i;
                    highest_outflow_sink_lon = j;
                }
            }
            delete working_coords;
        }
    }
    delete flow_direction_for_sink;
    main_sink_coords = new latlon_coords(highest_outflow_sink_lat,
                                         highest_outflow_sink_lon);
    return combined_cumulative_flow_to_sinks;
}

int latlon_cell::get_rmouth_outflow_combined_cumulative_flow(coords*&
                                                             main_outflow_coords) {
    int highest_outflow_rmouth_lat = 0;
    int highest_outflow_rmouth_lon = 0;
    int combined_cumulative_rmouth_outflow = 0;
    int greatest_flow_to_single_outflow = 0;
    direction_indicator* flow_direction_for_outflow = get_flow_direction_for_outflow();
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            int current_pixel_total_cumulative_flow =
                (*total_cumulative_flow)(working_coords);
            int current_pixel_flow_direction = (*river_directions)(working_coords);
            if (flow_direction_for_outflow->is_equal_to(current_pixel_flow_direction)) {
                combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow +
                    combined_cumulative_rmouth_outflow;
                if (greatest_flow_to_single_outflow <
                    current_pixel_total_cumulative_flow) {
                    greatest_flow_to_single_outflow =
                        current_pixel_total_cumulative_flow;
                    highest_outflow_rmouth_lat = i;
                    highest_outflow_rmouth_lon = j;
                }
            }
            delete working_coords;
        }
    }
    delete flow_direction_for_outflow;
    main_outflow_coords = new latlon_coords(highest_outflow_rmouth_lat,
                                            highest_outflow_rmouth_lon);
    return combined_cumulative_rmouth_outflow;
}

bool irregular_latlon_cell::check_if_coords_are_in_area(coords* coords_in) {
    int working_cell_number = (*cell_numbers)(coords_in);
    return (working_cell_number == cell_number);
}

bool irregular_latlon_neighborhood::
        check_if_coords_are_in_area(coords* coords_in) {
    bool within_area = false;
    int working_cell_number = (*cell_numbers)(coords_in);
    for (list<int>::iterator i = list_of_cells_in_neighborhood.begin();
         i != list_of_cells_in_neighborhood.end();i++){
        if(working_cell_number == *i) {
            within_area = true;
        }
    }
    return within_area;
}

int irregular_latlon_cell::get_sink_combined_cumulative_flow(coords*&
                                                             main_sink_coords) {
    int highest_outflow_sink_lat = 0;
    int highest_outflow_sink_lon = 0;
    int combined_cumulative_flow_to_sinks = 0;
    int greatest_flow_to_single_sink = 0;
    direction_indicator* flow_direction_for_sink = get_flow_direction_for_sink();
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            if ((*cell_numbers)(working_coords) == cell_number) {
                int current_pixel_total_cumulative_flow =
                    (*total_cumulative_flow)(working_coords);
                int current_pixel_flow_direction =
                    (*river_directions)(working_coords);
                if (flow_direction_for_sink->is_equal_to(current_pixel_flow_direction)) {
                    combined_cumulative_flow_to_sinks = current_pixel_total_cumulative_flow +
                        combined_cumulative_flow_to_sinks;
                    if (greatest_flow_to_single_sink <
                        current_pixel_total_cumulative_flow) {
                        greatest_flow_to_single_sink =
                            current_pixel_total_cumulative_flow;
                        highest_outflow_sink_lat = i;
                        highest_outflow_sink_lon = j;
                    }
                }
            }
            delete working_coords;
        }
    }
    delete flow_direction_for_sink;
    main_sink_coords = new latlon_coords(highest_outflow_sink_lat,
                                         highest_outflow_sink_lon);
    return combined_cumulative_flow_to_sinks;
}

int irregular_latlon_cell::get_rmouth_outflow_combined_cumulative_flow(coords*&
                                                                       main_outflow_coords){
    int highest_outflow_rmouth_lat = 0;
    int highest_outflow_rmouth_lon = 0;
    int combined_cumulative_rmouth_outflow = 0;
    int greatest_flow_to_single_outflow = 0;
    direction_indicator* flow_direction_for_outflow =
        get_flow_direction_for_outflow();
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            if ((*cell_numbers)(working_coords) == cell_number) {
                int current_pixel_total_cumulative_flow =
                    (*total_cumulative_flow)(working_coords);
                int current_pixel_flow_direction =
                    (*river_directions)(working_coords);
                if (flow_direction_for_outflow->is_equal_to(current_pixel_flow_direction)) {
                    combined_cumulative_rmouth_outflow = current_pixel_total_cumulative_flow +
                        combined_cumulative_rmouth_outflow;
                    if (greatest_flow_to_single_outflow <
                        current_pixel_total_cumulative_flow) {
                        greatest_flow_to_single_outflow =
                            current_pixel_total_cumulative_flow;
                        highest_outflow_rmouth_lat = i;
                        highest_outflow_rmouth_lon = j;
                    }
                }
            }
            delete working_coords;
        }
    }
    delete flow_direction_for_outflow;
    main_outflow_coords = new latlon_coords(highest_outflow_rmouth_lat,
                                            highest_outflow_rmouth_lon);
    return combined_cumulative_rmouth_outflow;
}


// void latlon_area::print_area(){
//     cout << "Area Information" << endl;
//     cout << "min lat=" << section_min_lat << endl;
//     cout << "min lon=" << section_min_lon << endl;
//     cout << "max lat=" << section_max_lat << endl;
//     cout << "max lon=" << section_max_lon << endl;
//     cout << "area rdirs" << endl;
//     for (int i = section_min_lat; i <= section_max_lat; i++) {
//         for( int j = section_min_lon; j <= section_max_lon; j++) {
//             cout << (*river_directions)(new latlon_coords(i,j));
//         }
//         cout << endl;
//     }
//     cout <<  "area total cumulative flow" << endl;
//     for (int i = section_min_lat; i <= section_max_lat; i++) {
//         for (int j = section_min_lon; j <= section_max_lon; j++) {
//             cout << (*total_cumulative_flow)(new latlon_coords(i,j));
//         }
//         cout << endl;
//     }
// }

dir_based_rdirs_latlon_cell::
    dir_based_rdirs_latlon_cell(latlon_section_coords* cell_section_coords_in,
                                grid_params* grid_params_in,
                                int* river_directions_in,
                                int* total_cumulative_flow_in,
                                rdir_upscaling_parameters* parameters_in) :
    latlon_cell(cell_section_coords_in,grid_params_in,
                river_directions_in,total_cumulative_flow_in,
                parameters_in){
    cell_neighborhood = new dir_based_rdirs_latlon_neighborhood(cell_section_coords_in,
                                                                grid_params_in,
                                                                river_directions_in,
                                                                total_cumulative_flow_in,
                                                                parameters_in);
}

dir_based_rdirs_irregular_latlon_cell::
    dir_based_rdirs_irregular_latlon_cell(irregular_latlon_section_coords* cell_section_coords_in,
                                          grid_params* grid_params_in,
                                          int* coarse_cell_neighbors_in,
                                          int* river_directions_in,
                                          int* total_cumulative_flow_in,
                                          int max_nneigh_coarse_in,
                                          rdir_upscaling_parameters* parameters_in) :
    irregular_latlon_cell(cell_section_coords_in,
                          grid_params_in,
                          river_directions_in,
                          total_cumulative_flow_in,
                          parameters_in) {
    cell_neighborhood = new dir_based_rdirs_irregular_latlon_neighborhood(cell_section_coords_in,
                                                                          grid_params_in,
                                                                          coarse_cell_neighbors_in,
                                                                          river_directions_in,
                                                                          total_cumulative_flow_in,
                                                                          max_nneigh_coarse_in,
                                                                          parameters_in);
}

// void latlon_cell::flow_calculate_river_directions_as_indices(int* coarse_river_direction_indices,
//                                                              section_coords*
//                                                              required_coarse_section_coords) {
//     int lat_offset;
//     int lon_offset;
//     if (required_coarse_section_coords) {
//         latlon_required_coarse_section_coords
//         int lat_offset = required_coarse_section_coords->section_min_lat - 1;
//         int lon_offset = required_coarse_section_coords->section_min_lon - 1;
//     } else {
//         lat_offset = 0;
//         lon_offset = 0;
//     }
//     int outlet_pixel_type;
//     coords* initial_outlet_pixel = flow_retrieve_initial_outlet_pixel(outlet_pixel_type)
//     coords* initial_cell_coords = cell_neighborhood->flow_get_cell_coords(initial_outlet_pixel)
//     if (outlet_pixel_type == flow_sink_outlet || outlet_pixel_type == flow_rmouth_outlet) {
//             coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),1) =
//                 -abs(outlet_pixel_type);
//             coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),2) =
//                 -abs(outlet_pixel_type);
//         delete initial_cell_coords;
//         delete initial_outlet_pixel;
//         return
//     }
//     coords* destination_cell_coords = cell_neighborhood->flow_find_downstream_cell(initial_outlet_pixel);
//     coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),1) =
//         destination_cell_coords->get_lat() - lat_offset;
//     coarse_river_direction_indices(initial_cell_coords->get_lat(),initial_cell_coords->get_lon(),2) =
//         destination_cell_coords->get_lon() - lon_offset;
//     delete initial_cell_coords;
//     delete destination_cell_coords;
//     delete initial_outlet_pixel;
// }

bool dir_based_rdirs_area::
     neighbor_flows_to_pixel(coords* coords_in,
                             direction_indicator*
                             flipped_direction_from_neighbor) {
    int directions_pointing_to_pixel[9] = { 9,8,7,6,5,4,3,2,1 };
    int rdir = (*river_directions)(coords_in);
    dir_based_direction_indicator* dir_based_flipped_direction_from_neighbor =
        static_cast<dir_based_direction_indicator*>(flipped_direction_from_neighbor);
    return (rdir == directions_pointing_to_pixel[
            dir_based_flipped_direction_from_neighbor->get_direction()-1]);
}

bool index_based_rdirs_area::neighbor_flows_to_pixel(
        coords* coords_in,
        direction_indicator* flipped_direction_from_neighbor) {
    //Dummy function; return true to satify return type requirements
    return true;
}


direction_indicator* dir_based_rdirs_latlon_cell::
    get_flow_direction_for_sink(bool is_for_cell) {
    return new dir_based_direction_indicator(flow_direction_for_sink);
}

direction_indicator* dir_based_rdirs_latlon_cell::
    get_flow_direction_for_outflow(bool is_for_cell) {
    return new dir_based_direction_indicator(flow_direction_for_outflow);
}

direction_indicator* dir_based_rdirs_latlon_cell::
    get_flow_direction_for_ocean_point(bool is_for_cell) {
    return new dir_based_direction_indicator(flow_direction_for_ocean_point);
}

direction_indicator* dir_based_rdirs_irregular_latlon_cell::
        get_flow_direction_for_sink(bool is_for_cell) {
    if (is_for_cell) {
        return new index_based_direction_indicator(cell_flow_direction_for_sink);
    } else {
        return new index_based_direction_indicator(flow_direction_for_sink);
    }
}

direction_indicator* dir_based_rdirs_irregular_latlon_cell::
        get_flow_direction_for_outflow(bool is_for_cell) {
    if (is_for_cell) {
        return new index_based_direction_indicator(cell_flow_direction_for_outflow);
    } else {
        return new index_based_direction_indicator(flow_direction_for_outflow);
    }
}

direction_indicator* dir_based_rdirs_irregular_latlon_cell::
        get_flow_direction_for_ocean_point(bool is_for_cell) {
    if (is_for_cell) {
        return new index_based_direction_indicator(cell_flow_direction_for_ocean_point);
    } else {
        return new index_based_direction_indicator(flow_direction_for_ocean_point);
    }
}

bool latlon_area::is_diagonal(coords* coords_in) {
    int rdir = (*river_directions)(coords_in);
    if (rdir == 7 || rdir == 9 || rdir == 1 || rdir == 3) {
        return true;
    } else {
        return false;
    }
}

void dir_based_rdirs_latlon_cell::mark_ocean_and_river_mouth_points() {
    bool non_ocean_cell_found = false;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            int rdir = (*river_directions)(working_coords);
            delete working_coords;
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

void dir_based_rdirs_irregular_latlon_cell::
        mark_ocean_and_river_mouth_points() {
    bool non_ocean_cell_found = false;
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* working_coords = new latlon_coords(i,j);
            if ((*cell_numbers)(working_coords) == cell_number) {
                int rdir = (*river_directions)(working_coords);
                if (rdir == flow_direction_for_outflow) {
                    contains_river_mouths = true;
                    ocean_cell = false;
                } else if (rdir != flow_direction_for_ocean_point) {
                    non_ocean_cell_found = true;
                }
                if( contains_river_mouths ) {
                    delete working_coords;
                    return;
                }
            }
            delete working_coords;
        }
    }
    contains_river_mouths = false;
    ocean_cell = ! non_ocean_cell_found;
}

int neighborhood::find_downstream_cell(coords* initial_outlet_pixel) {
    int initial_cumulative_flow = (*total_cumulative_flow)(initial_outlet_pixel);
    coords* working_pixel = initial_outlet_pixel->clone();
    bool coords_not_in_neighborhood = false;
    bool outflow_pixel_reached;
    working_pixel = find_next_pixel_downstream(working_pixel,
                                               coords_not_in_neighborhood,
                                               outflow_pixel_reached);
    int downstream_cell;
    while(true) {
        downstream_cell = in_which_cell(working_pixel);
        int last_cell_cumulative_flow;
        while(true) {
            last_cell_cumulative_flow = (*total_cumulative_flow)(working_pixel);
            working_pixel = find_next_pixel_downstream(working_pixel,
                                                       coords_not_in_neighborhood,
                                                       outflow_pixel_reached);
            if (coords_not_in_neighborhood ||
                downstream_cell != in_which_cell(working_pixel) ||
                outflow_pixel_reached) break;
        }
        if (coords_not_in_neighborhood || outflow_pixel_reached) break;
        if ((last_cell_cumulative_flow  - initial_cumulative_flow >
             parameters->area_threshold) &&
            ( ! is_center_cell(downstream_cell))) break;
    }
    delete working_pixel;
    return downstream_cell;
}

coords* neighborhood::flow_find_downstream_cell(coords* initial_outlet_pixel) {
    bool downstream_cell_found = false;
    coords* working_pixel = initial_outlet_pixel->clone();
    coords* cell_coords = flow_get_cell_coords(initial_outlet_pixel);
    double downstream_path_length = 0.0;
    int count_of_cells_passed_through = 0;
    while (true) {
        bool outflow_pixel_reached;
        bool coords_not_in_neighborhood;
        working_pixel = find_next_pixel_downstream(working_pixel,coords_not_in_neighborhood,
                                                   outflow_pixel_reached);
        downstream_path_length = downstream_path_length +
                                 calculate_length_through_pixel(working_pixel);
        if (coords_not_in_neighborhood) working_pixel = flow_wrap_coordinates(working_pixel);
        coords* new_cell_coords = flow_get_cell_coords(working_pixel);
        if ((*cell_coords) != (*new_cell_coords)) {
            delete cell_coords;
            cell_coords = new_cell_coords->clone();
            count_of_cells_passed_through = count_of_cells_passed_through + 1;
        }
        delete new_cell_coords;
        pixel_type flow_outlet_pixel_type =
            (*flow_outlet_pixels)(working_pixel);
        if (flow_outlet_pixel_type == flow_normal_outlet &&
            downstream_path_length > parameters->MUFP) downstream_cell_found = true;
        if (flow_outlet_pixel_type == flow_rmouth_outlet ||
            flow_outlet_pixel_type == flow_sink_outlet)
                downstream_cell_found = true;
        if (downstream_cell_found) break;
        if (count_of_cells_passed_through > parameters->flow_max_range) {
            delete working_pixel;
            working_pixel = initial_outlet_pixel->clone();
            working_pixel = find_next_pixel_downstream(working_pixel,
                                                       coords_not_in_neighborhood,
                                                       outflow_pixel_reached);
            if (coords_not_in_neighborhood) working_pixel = flow_wrap_coordinates(working_pixel);
            delete cell_coords;
            cell_coords = flow_get_cell_coords(working_pixel);
            break;
        }
    }
    delete working_pixel;
    return cell_coords;
}

bool neighborhood::path_reenters_region(coords*& coords_inout) {
    while(true) {
        bool coords_not_in_neighborhood;
        find_next_pixel_upstream(coords_inout,coords_not_in_neighborhood);
        if (coords_not_in_neighborhood) {
            return false;
        } else if (in_which_cell(coords_inout) == center_cell_id) {
            return true;
        }
    }
}

bool neighborhood::is_center_cell(int cell_id) {
    if (cell_id == center_cell_id) return true;
    else return false;
}

latlon_neighborhood::
    latlon_neighborhood(section_coords* center_cell_section_coords,
                        grid_params* grid_params_in,
                        int* river_directions_in,
                        int* total_cumulative_flow_in,
                        rdir_upscaling_parameters* parameters_in) {
    _grid_params = grid_params_in;
    if (parameters_in) {
        parameters = parameters_in;
        delete_parameters_on_exit = false;
    } else {
        parameters = new rdir_upscaling_parameters();
        delete_parameters_on_exit = true;
    }
    latlon_section_coords* latlon_center_cell_section_coords =
        static_cast<latlon_section_coords*>(center_cell_section_coords);
    center_cell_min_lat = latlon_center_cell_section_coords->section_min_lat;
    center_cell_min_lon = latlon_center_cell_section_coords->section_min_lon;
    center_cell_max_lat =  latlon_center_cell_section_coords->section_min_lat +
                                latlon_center_cell_section_coords->section_width_lat - 1;
    center_cell_max_lon =  latlon_center_cell_section_coords->section_min_lon +
                                latlon_center_cell_section_coords->section_width_lon - 1;
    section_min_lat = latlon_center_cell_section_coords->section_min_lat -
                           latlon_center_cell_section_coords->section_width_lat;
    section_min_lon = latlon_center_cell_section_coords->section_min_lon -
                           latlon_center_cell_section_coords->section_width_lon;
    section_max_lat = center_cell_max_lat +
                           latlon_center_cell_section_coords->section_width_lat;
    section_max_lon = center_cell_max_lon +
                           latlon_center_cell_section_coords->section_width_lon;
    neighborhood_section_coords =
        new latlon_section_coords(section_min_lat,section_min_lon,
                                  section_max_lat,section_max_lon);
    river_directions = new latlon_field_section<int>(river_directions_in,
                                                     neighborhood_section_coords,
                                                     grid_params_in);
    total_cumulative_flow = new latlon_field_section<int>(total_cumulative_flow_in,
                                                          neighborhood_section_coords,
                                                          grid_params_in);
    center_cell_id = 5;
}

irregular_latlon_neighborhood::
    irregular_latlon_neighborhood(section_coords* center_cell_section_coords,
                                  grid_params* grid_params_in,
                                  int* coarse_cell_neighbors_in,
                                  int* river_directions_in,
                                  int* total_cumulative_flow_in,
                                  int max_nneigh_coarse,
                                  rdir_upscaling_parameters* parameters_in) {
    _grid_params = grid_params_in;
    if (parameters_in) {
        parameters = parameters_in;
        delete_parameters_on_exit = false;
    } else {
        parameters = new rdir_upscaling_parameters();
        delete_parameters_on_exit = true;
    }
    irregular_latlon_section_coords* irreg_latlon_center_cell_section_coords =
        static_cast<irregular_latlon_section_coords*>(center_cell_section_coords);
    center_cell_id = irreg_latlon_center_cell_section_coords->cell_number;
    list_of_cells_in_neighborhood.clear();
    list_of_cells_in_neighborhood.push_back(center_cell_id);
    for (int i=0; i < max_nneigh_coarse; i++) {
        int nbr_cell_number = coarse_cell_neighbors_in[(center_cell_id-1)*max_nneigh_coarse+i];
        if (nbr_cell_number > 0) {
            list_of_cells_in_neighborhood.push_back(nbr_cell_number);
        }
    }
    neighborhood_section_coords =
        new irregular_latlon_section_coords(irreg_latlon_center_cell_section_coords->ncells,
                                            center_cell_id,
                                            irreg_latlon_center_cell_section_coords->cell_numbers,
                                            irreg_latlon_center_cell_section_coords->section_min_lats,
                                            irreg_latlon_center_cell_section_coords->section_min_lons,
                                            irreg_latlon_center_cell_section_coords->section_max_lats,
                                            irreg_latlon_center_cell_section_coords->section_max_lons );
    cell_numbers = new latlon_field_section<int>(irreg_latlon_center_cell_section_coords->cell_numbers,
                                                 neighborhood_section_coords,
                                                 grid_params_in);

    river_directions = new latlon_field_section<int>(river_directions_in,
                                                     neighborhood_section_coords,
                                                     grid_params_in);
    total_cumulative_flow = new latlon_field_section<int>(total_cumulative_flow_in,
                                                          neighborhood_section_coords,
                                                          grid_params_in);
}

// void flow_init_latlon_neighborhood(latlon_section_coords* center_cell_section_coords,
//                                        int* river_directions,
//                                        int* total_cumulative_flow,
//                                        int* flow_outlet_pixels_pointer,
//                                        latlon_section_coords* flow_section_coords) {
//     init_latlon_neighborhood(center_cell_section_coords,river_directions,
//                              total_cumulative_flow);
//     section_min_lat = flow_section_coords->section_min_lat;
//     section_min_lon = flow_section_coords->section_min_lon;
//     section_max_lat = flow_section_coords->section_min_lat +
//                       flow_section_coords->section_width_lat - 1;
//     section_max_lon = flow_section_coords->section_min_lon +
//                       flow_section_coords->section_width_lon - 1;
//     flow_outlet_pixels = latlon_field_section(flow_outlet_pixels_pointer,
//                                               flow_section_coords);
//     flow_cell_width_lat = center_cell_max_lat + 1 - center_cell_min_lat;
//     flow_cell_width_lon = center_cell_max_lon + 1 - center_cell_min_lon;
// }

int latlon_neighborhood::in_which_cell(coords* pixel) {
    int internal_cell_lat;
    int internal_cell_lon;
    latlon_coords* latlon_pixel = static_cast<latlon_coords*>(pixel);
    if (latlon_pixel->get_lat() < center_cell_min_lat) {
        internal_cell_lat = 1;
    } else if (latlon_pixel->get_lat() > center_cell_max_lat) {
        internal_cell_lat = 3;
    } else {
        internal_cell_lat = 2;
    }
    if (latlon_pixel->get_lon() > section_max_lon) {
        //Wrapped
        internal_cell_lon = 1;
    } else if (latlon_pixel->get_lon() < section_min_lon) {
        //Wrapped
        internal_cell_lon = 3;
    } else if (latlon_pixel->get_lon() < center_cell_min_lon) {
        internal_cell_lon = 1;
    } else if (latlon_pixel->get_lon() > center_cell_max_lon) {
        internal_cell_lon = 3;
    } else {
        internal_cell_lon = 2;
    }
    return internal_cell_lon + 9 - 3*internal_cell_lat;
}

int irregular_latlon_neighborhood::in_which_cell(coords* pixel) {
    return (*cell_numbers)(pixel);
}

coords* latlon_neighborhood::flow_get_cell_coords(coords* pixel_coords) {
    latlon_coords* latlon_pixel_coords = static_cast<latlon_coords*>(pixel_coords);
    return new latlon_coords(ceil((double)(latlon_pixel_coords->get_lat())/flow_cell_width_lat),
                             ceil((double)(latlon_pixel_coords->get_lon())/flow_cell_width_lon));
}


coords* irregular_latlon_neighborhood::flow_get_cell_coords(coords* pixel_coords) {
    //Dummy function
    return nullptr;
}

//This will need updating if we use neighborhood smaller than the full grid size for the
//flow-style algorithm - at the moment it assumes at neighborhood edges are edges of the
//full latitude-longitude grid and need longitudinal wrapping.
coords* latlon_neighborhood::flow_wrap_coordinates(coords* pixel_coords){
    if ( ! parameters->flow_wrap ) return pixel_coords;
    latlon_coords* latlon_pixel_coords = static_cast<latlon_coords*>(pixel_coords);
    int lon;
    if (latlon_pixel_coords->get_lon() < section_min_lon) {
        lon = section_max_lon + latlon_pixel_coords->get_lon() + 1 - section_min_lon;
    } else if (latlon_pixel_coords->get_lon() > section_max_lon) {
        lon = section_min_lon + latlon_pixel_coords->get_lon() - 1 - section_max_lon;
    } else {
        lon = latlon_pixel_coords->get_lon();
    }
    int lat = latlon_pixel_coords->get_lat();
    delete pixel_coords;
    return new latlon_coords(lat,lon);
}

// void flow_irregular_latlon_wrap_coordinates_dummy(coords* pixel_coords){}

coords* dir_based_rdirs_area::
     find_next_pixel_downstream(coords* coords_in,
                                bool& coords_not_in_area,
                                bool& outflow_pixel_reached) {
    int rdir = (*river_directions)(coords_in);
    int lat;
    int lon;
    latlon_coords* latlon_coords_in =
        static_cast<latlon_coords*>(coords_in);
    if ( rdir == 7 || rdir == 8 || rdir == 9) {
        lat = latlon_coords_in->get_lat() - 1;
    } else if ( rdir == 1 || rdir == 2 || rdir == 3) {
        lat = latlon_coords_in->get_lat() + 1;
    } else {
        lat = latlon_coords_in->get_lat();
    }
    if ( rdir == 7 || rdir == 4 || rdir == 1 ) {
        lon = latlon_coords_in->get_lon() - 1;
    } else if ( rdir == 9 || rdir == 6 || rdir == 3 ) {
        lon = latlon_coords_in->get_lon() + 1;
    } else {
        lon = latlon_coords_in->get_lon();
    }
    latlon_grid_params* _latlon_grid_params =
        static_cast<latlon_grid_params*>(_grid_params);
    if (!_latlon_grid_params->get_nowrap()) {
        if (lon < 0) {
            lon = lon + _latlon_grid_params->get_nlon();
        } else if (lon >= _latlon_grid_params->get_nlon()) {
            lon = lon - _latlon_grid_params->get_nlon();
        }
    }
    coords* working_coords = new latlon_coords(lat,lon);
    delete coords_in;
    if ( rdir == 5 || rdir == 0 || rdir == -1 ) {
        coords_not_in_area = false;
        outflow_pixel_reached =  true;
    } else {
        outflow_pixel_reached =  false;
        if (check_if_coords_are_in_area(working_coords)) {
            coords_not_in_area = false;
        } else {
            coords_not_in_area = true;
        }
    }
    return working_coords;
}

coords* index_based_rdirs_area::
    find_next_pixel_downstream(coords* coords_in,
                               bool& coords_not_in_area,
                               bool& outflow_pixel_reached) {
    int next_cell_index = (*river_directions)(coords_in);
    coords* coords_out = nullptr;
    if ( next_cell_index == index_for_sink ||
         next_cell_index == index_for_outflow ||
         next_cell_index == index_for_ocean_point ) {
        coords_not_in_area = false;
        outflow_pixel_reached =  true;
    } else {
        outflow_pixel_reached =  false;
        if (check_if_coords_are_in_area(coords_in)) {
            coords_not_in_area = false;
        } else {
            coords_not_in_area = true;
        }
    }
    delete coords_in;
    coords_out = new generic_1d_coords(next_cell_index);
    return coords_out;
}

dir_based_rdirs_latlon_neighborhood::
    dir_based_rdirs_latlon_neighborhood(latlon_section_coords* cell_section_coords_in,
                                        grid_params* grid_params_in,
                                        int* river_directions_in,
                                        int* total_cumulative_flow_in,
                                        rdir_upscaling_parameters* parameters_in) :
    latlon_neighborhood(cell_section_coords_in,
                        grid_params_in,
                        river_directions_in,
                        total_cumulative_flow_in,
                        parameters_in) {};

dir_based_rdirs_irregular_latlon_neighborhood::
    dir_based_rdirs_irregular_latlon_neighborhood(latlon_section_coords* cell_section_coords_in,
                                                  grid_params* grid_params_in,
                                                  int* coarse_cell_neighbors_in,
                                                  int* river_directions_in,
                                                  int* total_cumulative_flow_in,
                                                  int max_nneigh_coarse_in,
                                                  rdir_upscaling_parameters* parameters_in) :
    irregular_latlon_neighborhood(cell_section_coords_in,
                                  grid_params_in,
                                  coarse_cell_neighbors_in,
                                  river_directions_in,
                                  total_cumulative_flow_in,
                                  max_nneigh_coarse_in,
                                  parameters_in) {};

direction_indicator* dir_based_rdirs_latlon_neighborhood ::
    calculate_direction_indicator(int downstream_cell) {
    return new dir_based_direction_indicator(downstream_cell);
}

direction_indicator* dir_based_rdirs_irregular_latlon_neighborhood::
    calculate_direction_indicator(int downstream_cell) {
    return new index_based_direction_indicator(downstream_cell);
}

latlon_extended_area::latlon_extended_area(section_coords* area_section_coords_in,
                                           grid_params* area_grid_params_in,
                                           int* river_directions_in) {
        _grid_params = area_grid_params_in;
        latlon_section_coords* latlon_area_section_coords =
            static_cast<latlon_section_coords*>(area_section_coords_in);
        section_min_lat = latlon_area_section_coords->section_min_lat;
        section_min_lon = latlon_area_section_coords->section_min_lon;
        section_max_lat = latlon_area_section_coords->section_min_lat +
                          latlon_area_section_coords->section_width_lat - 1;
        section_max_lon =  latlon_area_section_coords->section_min_lon +
                           latlon_area_section_coords->section_width_lon - 1;
        total_cumulative_flow = nullptr;
        river_directions = new field_section<int>(river_directions_in,
                                                  area_grid_params_in);
        cells_to_reprocess = new field<bool>(area_grid_params_in);
        cells_to_reprocess->switch_data_deletion_off();
}

icon_single_index_extended_area::icon_single_index_extended_area(section_coords* area_section_coords_in,
                                                                 grid_params* area_grid_params_in,
                                                                 int* river_directions_in) {
    _grid_params = area_grid_params_in;
    ncells = static_cast<icon_single_index_grid_params*>
                (area_grid_params_in)->get_ncells();
    total_cumulative_flow = nullptr;
    river_directions = new field_section<int>(river_directions_in,
                                              area_grid_params_in);
    cells_to_reprocess = new field<bool>(area_grid_params_in);
    cells_to_reprocess->switch_data_deletion_off();
}

bool extended_area::check_cell_for_localized_loops(coords* coords_in) {
    coords* initial_cell_coords = coords_in->clone();
    coords* working_coords = coords_in->clone();
    bool coords_not_in_area;
    bool outflow_pixel_reached;
    working_coords =
        find_next_pixel_downstream(working_coords,coords_not_in_area,
                                   outflow_pixel_reached);
    if (outflow_pixel_reached) {
        delete working_coords;
        delete initial_cell_coords;
        return false;
    }
    bool is_sea_point_or_horizontal_edge_or_sink;
    if (*working_coords == *initial_cell_coords) {
        is_sea_point_or_horizontal_edge_or_sink = true;
    } else {
        is_sea_point_or_horizontal_edge_or_sink = false;
    }
    working_coords =
        find_next_pixel_downstream(working_coords,coords_not_in_area,
                                   outflow_pixel_reached);
    if (outflow_pixel_reached) {
        delete working_coords;
        delete initial_cell_coords;
        return false;
    }
    bool localized_loop_found;
    if (*working_coords == *initial_cell_coords &&
        ! is_sea_point_or_horizontal_edge_or_sink) {
        localized_loop_found = true;
    } else {
        localized_loop_found = false;
    }
    delete working_coords;
    delete initial_cell_coords;
    return localized_loop_found;
};

bool* latlon_extended_area::check_area_for_localized_loops(){
    cells_to_reprocess->set_all(false);
    for (int j = section_min_lon; j <= section_max_lon; j++) {
        for (int i = section_min_lat; i <= section_max_lat; i++) {
            coords* cell_coords = new latlon_coords(i,j);
            if (check_cell_for_localized_loops(cell_coords)) {
                (*cells_to_reprocess)(cell_coords) = true;
            }
            delete cell_coords;
        }
    }
    return cells_to_reprocess->get_array();
};

bool* icon_single_index_extended_area::check_area_for_localized_loops(){
    cells_to_reprocess->set_all(false);
    for(int i = 1; i <= ncells; i++) {
        coords* cell_coords = new generic_1d_coords(i);
        if (check_cell_for_localized_loops(cell_coords)) {
            (*cells_to_reprocess)(cell_coords) = true;
        }
        delete cell_coords;
    }
    return cells_to_reprocess->get_array();
};
